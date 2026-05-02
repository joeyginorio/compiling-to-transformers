import torch
import random
import string
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset

SEP = "|"
LETTERS = list(string.ascii_lowercase)
VOWELS = set("aeiou")
SEQ_LEN = 15
SEED = 42

VOCAB = LETTERS + [SEP]
encode = {tok: i for i, tok in enumerate(VOCAB)}
decode = {i: tok for tok, i in encode.items()}

def generate_example(rng: random.Random) -> str:
    chars = rng.choices(LETTERS, k=SEQ_LEN)
    output = chars[::-1] if chars[0] in VOWELS else chars
    return " ".join(chars + [SEP] + output)

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

def generate_example_fixed(rng: random.Random, transform, first_pool: list[str]) -> str:
    first = rng.choice(first_pool)
    rest = rng.choices(LETTERS, k=SEQ_LEN - 1)
    chars = [first] + rest
    output = transform(chars)
    return " ".join(chars + [SEP] + output)

def generate_datasets_balanced(out_dir: Path = Path("experiments/cond_reverse/data/dataset")) -> None:
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    consonants = [c for c in LETTERS if c not in VOWELS]
    vowels = [c for c in LETTERS if c in VOWELS]
    identity = [generate_example_fixed(rng, lambda cs: cs, consonants) for _ in range(2500)]
    reverse = [generate_example_fixed(rng, lambda cs: cs[::-1], vowels) for _ in range(2500)]
    (out_dir / "steer.txt").write_text("\n".join(reverse + identity) + "\n")

def generate_datasets(save=False, out_dir: Path = Path("experiments/cond_reverse/data/dataset")) -> dict[str, ReverseDataset]:
    rng = random.Random(SEED)
    splits = {"train": 10_000, "probe": 5_000, "val": 1_000, "test": 1_000}
    result = {}
    for name, n in splits.items():
        examples = generate_split(n, rng)
        if save:
            (out_dir / f"{name}.txt").write_text("\n".join(examples) + "\n")
        result[name] = ReverseDataset(examples)
    return result

datasets = generate_datasets()
