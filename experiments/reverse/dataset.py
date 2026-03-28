import torch
import random
import string
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset

SEP = "|"
LETTERS = list(string.ascii_lowercase)
SEQ_LEN = 15
SEED = 42

VOCAB = LETTERS + [SEP]
encode = {tok: i for i, tok in enumerate(VOCAB)}
decode = {i: tok for tok, i in encode.items()}

def generate_example(rng: random.Random) -> str:
    chars = rng.choices(LETTERS, k=SEQ_LEN)
    return " ".join(chars + [SEP] + chars[::-1])

def generate_split(n: int, rng: random.Random) -> list[str]:
    return [generate_example(rng) for _ in range(n)]

def tokenize(example: str) -> Tensor:
    return torch.tensor([encode[tok] for tok in example.split()], dtype=torch.long)

class ReverseDataset(Dataset):
    def __init__(self, examples: list[str]):
        self.tokens = [tokenize(ex) for ex in examples]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        toks = self.tokens[idx]
        return toks[:-1], toks[1:]

def generate_datasets(out_dir: Path = Path("experiments/reverse/data/dataset")) -> dict[str, ReverseDataset]:
    rng = random.Random(SEED)
    splits = {"train": 10_000, "val": 1_000, "test": 1_000}
    result = {}
    for name, n in splits.items():
        examples = generate_split(n, rng)
        (out_dir / f"{name}.txt").write_text("\n".join(examples) + "\n")
        result[name] = ReverseDataset(examples)
    return result

datasets = generate_datasets()
