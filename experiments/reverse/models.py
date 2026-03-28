import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.reverse.dataset import encode, decode, generate_datasets


def interpret(logits: torch.Tensor) -> list[str]:
    return [decode[int(i.item())] for i in logits.squeeze(0).argmax(dim=-1)]


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
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(x) + self.pos_emb(torch.arange(self.seq_len, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len, device=x.device)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        return self.out(h)

