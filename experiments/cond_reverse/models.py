import torch
import torch.nn as nn
from experiments.cond_reverse.dataset import decode, SEQ_LEN
import cajal.compiling as cj
from cajal.typing import check
from cajal.syntax import TmProd, TmProj, TmVar, TySum, TyUnit, TyProd, TmDict, TmLet, TmLookup, TmInj, TmUnit

# --- Program ---

elem_ty = TySum([TyUnit()] * 4)
reverse = TmProd([TmProj(14 - i, TmVar('x')) for i in range(15)])
check(reverse, {'x': TyProd([elem_ty] * 15)})
# Dictionary-based reverse: compiles to a linear attention head.
# Keys and values form a KV memory (position -> reversed element),
# and each lookup q attends via dot product: ∑_i ⟨k_i, q⟩ · v_i.
pos_ty = TySum([TyUnit()] * 15)
_keys = TmProd([TmInj(i, TmUnit(), pos_ty) for i in range(15)])
_vals = TmProd([TmProj(14 - i, TmVar('x')) for i in range(15)])
reverse_attn = TmLet(
    'd',
    TmDict(_keys, _vals),
    TmProd([
        TmLookup(TmVar('d'), TmInj(i, TmUnit(), pos_ty), lambda a, b: a == b)
        for i in range(15)
    ])
)
check(reverse_attn, {'x': TyProd([elem_ty] * 15)})

# NOTE: Set the programmed module here.
program_module = cj.compile(reverse_attn)

# --- Models ---

class ModelD(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 16, n_heads: int = 4, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.out = nn.Linear(d_model, vocab_size)

        d_head = d_model // n_heads
        self.prog = program_module
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, torch.Tensor] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        prog_in = self.prog_proj_in(x1[:, :SEQ_LEN, :]) # NOTE: Only pass in the first 15 chars!
        if steer is not None:
            for pos, vec in steer.items():
                prog_in[:, pos, :] = prog_in[:, pos, :] + vec
        prog_out = torch.vmap(self.prog)({'x': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r2 = self.mlp(r1) + r1
        logits = self.out(r2)

        if representations:
            prog_in_padded = torch.cat([torch.zeros_like(prog_in), prog_in], dim=1)
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "prog_in": prog_in_padded, "prog_out": prog_out_padded}
        else:
            return logits

class ModelU(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 16, n_heads: int = 4, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.out = nn.Linear(d_model, vocab_size)

        d_head = d_model // n_heads
        self.prog = cj.to_multilinear(program_module, rand=False, sample_env={'x': torch.zeros(SEQ_LEN * d_head)})
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, torch.Tensor] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        prog_in = self.prog_proj_in(x1[:, :SEQ_LEN, :])
        if steer is not None:
            for pos, vec in steer.items():
                prog_in[:, pos, :] = prog_in[:, pos, :] + vec
        prog_out = torch.vmap(self.prog)({'x': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r2 = self.mlp(r1) + r1
        logits = self.out(r2)

        if representations:
            prog_in_padded = torch.cat([torch.zeros_like(prog_in), prog_in], dim=1)
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "prog_in": prog_in_padded, "prog_out": prog_out_padded}
        else:
            return logits

class ModelT(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 16, n_heads: int = 4, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.out = nn.Linear(d_model, vocab_size)

        d_head = d_model // n_heads
        self.prog = cj.to_multilinear(program_module, rand=True, sample_env={'x': torch.zeros(SEQ_LEN * d_head)})
        self.prog_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False, 'prog_out': False},
                steer: dict[int, torch.Tensor] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        prog_in = self.prog_proj_in(x1[:, :SEQ_LEN, :])
        if steer is not None:
            for pos, vec in steer.items():
                prog_in[:, pos, :] = prog_in[:, pos, :] + vec
        prog_out = torch.vmap(self.prog)({'x': prog_in.flatten(start_dim=1)}).reshape(prog_in.shape)
        if ablate['prog_out']:
            prog_out = torch.roll(prog_out, shifts=1, dims=0)
        prog_out_padded = torch.cat([torch.zeros_like(prog_out), prog_out], dim=1)

        r1 = self.merge(torch.cat([h1, prog_out_padded], dim=-1)) + x1
        r2 = self.mlp(r1) + r1
        logits = self.out(r2)

        if representations:
            prog_in_padded = torch.cat([torch.zeros_like(prog_in), prog_in], dim=1)
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "prog_in": prog_in_padded, "prog_out": prog_out_padded}
        else:
            return logits

class ModelI(nn.Module):
    def __init__(self, vocab_size: int = 27, d_model: int = 16, n_heads: int = 4, seq_len: int = 30):
        super().__init__()
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(seq_len))
        self.register_buffer('positions', torch.arange(seq_len))

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                representations: bool = False,
                ablate: dict[str, bool] = {'h1': False}) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)

        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        if ablate['h1']:
            h1 = torch.roll(h1, shifts=1, dims=0)

        r1 = h1 + x1
        r2 = self.mlp(r1) + r1
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