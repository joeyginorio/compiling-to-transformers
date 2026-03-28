import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.reverse.dataset import decode

# --- Models ---

class ModelD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ModelT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.tok_emb(x) + self.pos_emb(self.positions)
        h1, _ = self.attn(x1, x1, x1, attn_mask=self.mask, is_causal=True, need_weights=False)
        r1 = h1 + x1
        r2 = self.mlp(r1) + r1
        return self.out(r2)
    

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