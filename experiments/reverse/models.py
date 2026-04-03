import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.reverse.dataset import decode
import cajal.compiling as cj
from cajal.typing import check
from cajal.syntax import TmProd, TmProj, TmVar, TySum, TyUnit, TyProd

# --- Program ---

elem_ty = TySum([TyUnit()] * 4)
reverse = TmProd([TmProj(29 - i, TmVar('x')) for i in range(30)])
check(reverse, {'x': TyProd([elem_ty] * 30)})
reverse_module = cj.compile(reverse)

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
        self.rev = reverse_module
        self.rev_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, representations: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)
        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        rev_in = self.rev_proj_in(x1)
        rev_out = torch.vmap(self.rev)({'x': rev_in.flatten(start_dim=1)}).reshape(rev_in.shape)
        r1 = self.merge(torch.cat([h1, rev_out], dim=-1)) + x1
        r2 = self.mlp(r1) + r1
        logits = self.out(r2)

        if representations:
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "rev_in": rev_in, "rev_out": rev_out}
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
        self.rev = cj.to_multilinear(reverse_module, rand=False, sample_env={'x': torch.zeros(seq_len * d_head)})
        self.rev_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, representations: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)
        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        rev_in = self.rev_proj_in(x1)
        rev_out = torch.vmap(self.rev)({'x': rev_in.flatten(start_dim=1)}).reshape(rev_in.shape)
        r1 = self.merge(torch.cat([h1, rev_out], dim=-1)) + x1
        r2 = self.mlp(r1) + r1
        logits = self.out(r2)

        if representations:
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "rev_in": rev_in, "rev_out": rev_out}
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
        self.rev = cj.to_multilinear(reverse_module, rand=True, sample_env={'x': torch.zeros(seq_len * d_head)})
        self.rev_proj_in = nn.Linear(d_model, d_head, bias=False)
        self.merge = nn.Linear(d_model + d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, representations: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)
        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        rev_in = self.rev_proj_in(x1)
        rev_out = torch.vmap(self.rev)({'x': rev_in.flatten(start_dim=1)}).reshape(rev_in.shape)
        r1 = self.merge(torch.cat([h1, rev_out], dim=-1)) + x1
        r2 = self.mlp(r1) + r1
        logits = self.out(r2)

        if representations:
            return logits, {"x1": x1, "h1": h1, "r1": r1, "r2": r2, "rev_in": rev_in, "rev_out": rev_out}
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

    def forward(self, x: torch.Tensor, representations: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)
        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
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