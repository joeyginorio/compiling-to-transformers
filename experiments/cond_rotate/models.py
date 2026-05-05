import torch
import torch.nn as nn
from experiments.cond_rotate.dataset import decode, SEQ_LEN
import cajal.compiling as cj
from cajal.typing import check
from cajal.syntax import Tm, TmProd, TmProj, TmVar, TySum, TyUnit, TyProd, TmDict, TmLet, TmLookup, TmInj, TmUnit, TmCase, TmSeq

# --- Programs ---

K = 15
N = 15
CHAR = TySum([TyUnit()] * K)
POS = TySum([TyUnit()] * N)
SEQ = TyProd([CHAR] * N)
VOWEL_BRANCH = 0

# Dictionary-based first-token routed rotation: compiles to a linear attention
# stack. Keys index positions, values are projections of `xs_value`, and each
# slot's lookup uses the first-token control as the query.
def rotate(x_ctrl_var: str, xs_value_var: str, direction: int) -> Tm:
    return TmLet(
        "d",
        TmDict(
            TmProd([TmInj(i, TmUnit(), POS) for i in range(N)]),
            TmProd([TmProj(i, TmVar(xs_value_var)) for i in range(N)]),
        ),
        TmProd([
            TmLookup(
                TmVar("d"),
                TmVar(x_ctrl_var),
                lambda key, query, slot=slot, direction=direction: key.n == (slot + direction * query.n) % N,
            )
            for slot in range(N)
        ]),
    )

rotate_attn: Tm = rotate("x_ctrl", "xs_value", 1)
right_rotate_attn: Tm = rotate("x_ctrl", "xs_value", -1)
check(rotate_attn, {"x_ctrl": CHAR, "xs_value": SEQ})
check(right_rotate_attn, {"x_ctrl": CHAR, "xs_value": SEQ})

# Conditional rotate: if the final element is represented by the learned
# "vowel" branch, left-rotate; otherwise right-rotate by the same first-token
# shift. `x_cond` is the extra linear copy used only for branch selection.
case_vars = [f"_v{i}" for i in range(K)]
branches: list[Tm] = [
    TmSeq(TmVar(case_vars[i]), rotate_attn if i == VOWEL_BRANCH else right_rotate_attn)
    for i in range(K)
]
cond_rotate_attn = TmCase(
    TmVar("x_cond"),
    case_vars,
    branches,
)
check(cond_rotate_attn, {"x_cond": CHAR, "x_ctrl": CHAR, "xs_value": SEQ})

# NOTE: Set the programmed module here.
program_module = cj.compile(cond_rotate_attn)


class CondRotateMultilinear(nn.Module):
    def __init__(self, d_head: int, rand: bool = False):
        super().__init__()
        if d_head != K:
            raise ValueError(f"cond_rotate expects d_head == {K}, got {d_head}")

        d_value = N * d_head
        self.lmap = nn.Linear(d_head * d_head * d_value, d_value, bias=False)
        if not rand:
            weight = torch.zeros(d_value, d_head * d_head * d_value)
            for cond_branch in range(d_head):
                direction = 1 if cond_branch == VOWEL_BRANCH else -1
                for ctrl_branch in range(d_head):
                    for slot in range(N):
                        value_pos = (slot + direction * ctrl_branch) % N
                        for channel in range(d_head):
                            out_idx = slot * d_head + channel
                            value_idx = value_pos * d_head + channel
                            in_idx = ((cond_branch * d_head + ctrl_branch) * d_value) + value_idx
                            weight[out_idx, in_idx] = 1.0
            with torch.no_grad():
                self.lmap.weight.copy_(weight)

    def forward(self, env: dict[str, torch.Tensor]) -> torch.Tensor:
        selector = torch.kron(env["x_cond"], env["x_ctrl"])
        return self.lmap(torch.kron(selector, env["xs_value"]))

# --- Models ---

class ModelF(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 30, n_heads: int = 2, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        d_head = d_model // n_heads
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.prog = program_module
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, tuple[float, float]] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        h1, _ = self.attn1(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        prog_in = self.prog_proj_in(x1[:, :SEQ_LEN, :])
        prog_in_sel = prog_in.clone()
        if steer is not None:
            for pos, (alpha, beta) in steer.items():
                prog_in_sel[:, pos, 0]  = prog_in_sel[:, pos, 0]  * alpha
                prog_in_sel[:, pos, 1:] = prog_in_sel[:, pos, 1:] * beta
        prog_out = torch.vmap(self.prog)({'x_cond': prog_in_sel[:, -1, :],
                                          'x_ctrl': prog_in_sel[:, 0, :],
                                          'xs_value': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r1 = self.mlp1(r1) + r1

        h2, _ = self.attn2(r1, r1, r1, attn_mask=self.mask, is_causal=True, need_weights=False)
        r2 = h2 + r1
        r2 = self.mlp2(r2) + r2

        logits = self.out(r2)

        if representations:
            prog_in_padded = torch.cat([torch.zeros_like(prog_in), prog_in], dim=1)
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "prog_in": prog_in_padded, "prog_out": prog_out_padded}
        else:
            return logits

class ModelU(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 30, n_heads: int = 2, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        d_head = d_model // n_heads
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.prog = CondRotateMultilinear(d_head, rand=False)
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, tuple[float, float]] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        h1, _ = self.attn1(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        prog_in = self.prog_proj_in(x1[:, :SEQ_LEN, :])
        prog_in_sel = prog_in.clone()
        if steer is not None:
            for pos, (alpha, beta) in steer.items():
                prog_in_sel[:, pos, 0]  = prog_in_sel[:, pos, 0]  * alpha
                prog_in_sel[:, pos, 1:] = prog_in_sel[:, pos, 1:] * beta
        prog_out = torch.vmap(self.prog)({'x_cond': prog_in_sel[:, -1, :],
                                          'x_ctrl': prog_in_sel[:, 0, :],
                                          'xs_value': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r1 = self.mlp1(r1) + r1

        h2, _ = self.attn2(r1, r1, r1, attn_mask=self.mask, is_causal=True, need_weights=False)
        r2 = h2 + r1
        r2 = self.mlp2(r2) + r2

        logits = self.out(r2)

        if representations:
            prog_in_padded = torch.cat([torch.zeros_like(prog_in), prog_in], dim=1)
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "prog_in": prog_in_padded, "prog_out": prog_out_padded}
        else:
            return logits



class ModelT(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 30, n_heads: int = 2, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        d_head = d_model // n_heads
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.prog = CondRotateMultilinear(d_head, rand=True)
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, tuple[float, float]] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        h1, _ = self.attn1(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        prog_in = self.prog_proj_in(x1[:, :SEQ_LEN, :])
        prog_in_sel = prog_in.clone()
        if steer is not None:
            for pos, (alpha, beta) in steer.items():
                prog_in_sel[:, pos, 0]  = prog_in_sel[:, pos, 0]  * alpha
                prog_in_sel[:, pos, 1:] = prog_in_sel[:, pos, 1:] * beta
        prog_out = torch.vmap(self.prog)({'x_cond': prog_in_sel[:, -1, :],
                                          'x_ctrl': prog_in_sel[:, 0, :],
                                          'xs_value': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r1 = self.mlp1(r1) + r1

        h2, _ = self.attn2(r1, r1, r1, attn_mask=self.mask, is_causal=True, need_weights=False)
        r2 = h2 + r1
        r2 = self.mlp2(r2) + r2

        logits = self.out(r2)

        if representations:
            prog_in_padded = torch.cat([torch.zeros_like(prog_in), prog_in], dim=1)
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "prog_in": prog_in_padded, "prog_out": prog_out_padded}
        else:
            return logits

class ModelI(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 30, n_heads: int = 2, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False}) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        h1, _ = self.attn1(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)
        r1 = h1 + x1
        r1 = self.mlp1(r1) + r1

        h2, _ = self.attn2(r1, r1, r1, attn_mask=self.mask, is_causal=True, need_weights=False)
        r2 = h2 + r1
        r2 = self.mlp2(r2) + r2

        logits = self.out(r2)

        if representations:
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2}
        else:
            return logits


# --- Helpers ---

def interpret_in(tokens: torch.Tensor) -> list[str]:
    tokens_list = list(tokens)
    result = []
    for seq in tokens_list:
        result.append(''.join([decode[int(i.item())] for i in seq])[:15])
    return result

def interpret_out(logits: torch.Tensor) -> list[str]:
    ys = [''.join(decode[i] for i in row.tolist()) for row in logits.argmax(dim=-1)]
    return [y[15:] for y in ys]
