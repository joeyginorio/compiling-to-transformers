import torch
import torch.nn as nn
from experiments.rotate.dataset import decode, SEQ_LEN
import cajal.compiling as cj
from cajal.typing import check
from cajal.syntax import Tm, TmProd, TmProj, TmVar, TySum, TyUnit, TyProd, TmDict, TmLet, TmLookup, TmInj, TmUnit

# --- Programs ---

K = 15
N = 15
CHAR = TySum([TyUnit()] * K)
POS = TySum([TyUnit()] * N)
SEQ = TyProd([CHAR] * N)

# Dictionary-based first-token routed rotation: compiles to a linear attention
# stack. Keys index positions, values are projections of `xs_value`, and each
# slot's lookup uses `xs_ctrl[0]` as the query with the per-slot relation
# `key.n == (slot + query.n) mod N`.
rotate_attn: Tm = TmLet(
    "d",
    TmDict(
        TmProd([TmInj(i, TmUnit(), POS) for i in range(N)]),
        TmProd([TmProj(i, TmVar("xs_value")) for i in range(N)]),
    ),
    TmProd([
        TmLookup(
            TmVar("d"),
            TmProj(0, TmVar("xs_ctrl")),
            lambda key, query, slot=slot: key.n == (slot + query.n) % N,
        )
        for slot in range(N)
    ]),
)
check(rotate_attn, {"xs_ctrl": SEQ, "xs_value": SEQ})

# NOTE: Set the programmed module here.
program_module = cj.compile(rotate_attn)

# --- Models ---

class ModelF(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 30, n_heads: int = 2, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        d_head = d_model // n_heads
        self.norm_attn1 = nn.RMSNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_prog = nn.RMSNorm(d_model)
        self.prog = program_module
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)
        self.norm_mlp1 = nn.RMSNorm(d_model)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_attn2 = nn.RMSNorm(d_model)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp2 = nn.RMSNorm(d_model)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_out = nn.RMSNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, tuple[float, float]] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        x1_n = self.norm_attn1(x1)
        h1, _ = self.attn1(x1_n, x1_n, x1_n, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        x1_pn = self.norm_prog(x1)
        prog_in = self.prog_proj_in(x1_pn[:, :SEQ_LEN, :])
        prog_in_sel = prog_in.clone()
        if steer is not None:
            for pos, (alpha, beta) in steer.items():
                prog_in_sel[:, pos, 0]  = prog_in_sel[:, pos, 0]  * alpha
                prog_in_sel[:, pos, 1:] = prog_in_sel[:, pos, 1:] * beta
        prog_out = torch.vmap(self.prog)({'xs_ctrl': prog_in_sel.flatten(start_dim=1),
                                          'xs_value': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r1 = self.mlp1(self.norm_mlp1(r1)) + r1

        r1_n = self.norm_attn2(r1)
        h2, _ = self.attn2(r1_n, r1_n, r1_n, attn_mask=self.mask, is_causal=True, need_weights=False)
        r2 = h2 + r1
        r2 = self.mlp2(self.norm_mlp2(r2)) + r2

        logits = self.out(self.norm_out(r2))

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
        self.norm_attn1 = nn.RMSNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_prog = nn.RMSNorm(d_model)
        self.prog = cj.to_multilinear(program_module, rand=False,
                                      sample_env={'xs_ctrl': torch.zeros(SEQ_LEN * d_head),
                                                  'xs_value': torch.zeros(SEQ_LEN * d_head)})
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)
        self.norm_mlp1 = nn.RMSNorm(d_model)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_attn2 = nn.RMSNorm(d_model)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp2 = nn.RMSNorm(d_model)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_out = nn.RMSNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, tuple[float, float]] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        x1_n = self.norm_attn1(x1)
        h1, _ = self.attn1(x1_n, x1_n, x1_n, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        x1_pn = self.norm_prog(x1)
        prog_in = self.prog_proj_in(x1_pn[:, :SEQ_LEN, :])
        prog_in_sel = prog_in.clone()
        if steer is not None:
            for pos, (alpha, beta) in steer.items():
                prog_in_sel[:, pos, 0]  = prog_in_sel[:, pos, 0]  * alpha
                prog_in_sel[:, pos, 1:] = prog_in_sel[:, pos, 1:] * beta
        prog_out = torch.vmap(self.prog)({'xs_ctrl': prog_in_sel.flatten(start_dim=1),
                                          'xs_value': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r1 = self.mlp1(self.norm_mlp1(r1)) + r1

        r1_n = self.norm_attn2(r1)
        h2, _ = self.attn2(r1_n, r1_n, r1_n, attn_mask=self.mask, is_causal=True, need_weights=False)
        r2 = h2 + r1
        r2 = self.mlp2(self.norm_mlp2(r2)) + r2

        logits = self.out(self.norm_out(r2))

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
        self.norm_attn1 = nn.RMSNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_prog = nn.RMSNorm(d_model)
        self.prog = cj.to_multilinear(program_module, rand=True,
                                      sample_env={'xs_ctrl': torch.zeros(SEQ_LEN * d_head),
                                                  'xs_value': torch.zeros(SEQ_LEN * d_head)})
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)
        self.norm_mlp1 = nn.RMSNorm(d_model)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_attn2 = nn.RMSNorm(d_model)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp2 = nn.RMSNorm(d_model)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_out = nn.RMSNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, tuple[float, float]] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        x1_n = self.norm_attn1(x1)
        h1, _ = self.attn1(x1_n, x1_n, x1_n, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        x1_pn = self.norm_prog(x1)
        prog_in = self.prog_proj_in(x1_pn[:, :SEQ_LEN, :])
        prog_in_sel = prog_in.clone()
        if steer is not None:
            for pos, (alpha, beta) in steer.items():
                prog_in_sel[:, pos, 0]  = prog_in_sel[:, pos, 0]  * alpha
                prog_in_sel[:, pos, 1:] = prog_in_sel[:, pos, 1:] * beta
        prog_out = torch.vmap(self.prog)({'xs_ctrl': prog_in_sel.flatten(start_dim=1),
                                          'xs_value': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r1 = self.mlp1(self.norm_mlp1(r1)) + r1

        r1_n = self.norm_attn2(r1)
        h2, _ = self.attn2(r1_n, r1_n, r1_n, attn_mask=self.mask, is_causal=True, need_weights=False)
        r2 = h2 + r1
        r2 = self.mlp2(self.norm_mlp2(r2)) + r2

        logits = self.out(self.norm_out(r2))

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

        self.norm_attn1 = nn.RMSNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp1 = nn.RMSNorm(d_model)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_attn2 = nn.RMSNorm(d_model)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp2 = nn.RMSNorm(d_model)
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_out = nn.RMSNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False}) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        x1_n = self.norm_attn1(x1)
        h1, _ = self.attn1(x1_n, x1_n, x1_n, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)
        r1 = h1 + x1
        r1 = self.mlp1(self.norm_mlp1(r1)) + r1

        r1_n = self.norm_attn2(r1)
        h2, _ = self.attn2(r1_n, r1_n, r1_n, attn_mask=self.mask, is_causal=True, need_weights=False)
        r2 = h2 + r1
        r2 = self.mlp2(self.norm_mlp2(r2)) + r2

        logits = self.out(self.norm_out(r2))

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
