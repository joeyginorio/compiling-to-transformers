from dataclasses import dataclass, field
from pathlib import Path
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from experiments.reverse.dataset import datasets, tokenize, SEQ_LEN
from experiments.reverse.models import interpret_out


# --- Globals ---

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

_data_dir = Path("experiments/reverse/data/learning")


# --- Measurements ---

_probes = [
    "a b c d e f g h i j k l m n o | o n m l k j i h g f e d c b a",
    "o n m l k j i h g f e d c b a | a b c d e f g h i j k l m n o",
]
_probes = torch.stack([tokenize(s)[:-1] for s in _probes]).to(DEVICE)

@dataclass
class Measure:
    model_name: str
    probes: torch.Tensor = field(default=_probes)
    records: list[dict] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)

    _every: int = field(default=50)

    def record(self, loss: float, model: nn.Module, loader: DataLoader,
               step: int, seed: int, lr: float, batch_size: int) -> None:
        if step % self._every == 0:
            val_loss = evaluate(model, loader)
            probe_outputs = self.run_probes(model)
            self.records.append({"step": step,
                                 "seed": seed,
                                 "lr": lr,
                                 "batch_size": batch_size,
                                 "train_loss": loss,
                                 "val_loss": val_loss,
                                 "probe_outputs": probe_outputs})
            print(f"step {step:5d}  train {loss:.4f}  val {val_loss:.4f}")

        if step % (self._every * 4) == 0:
            self.checkpoints.append({
                "step": step,
                "seed": seed,
                "lr": lr,
                "batch_size": batch_size,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            })

    def run_probes(self, model: nn.Module) -> list[str]:
        training = model.training
        model.eval()
        with torch.no_grad():
            result = interpret_out(model(self.probes))
        model.train(training)
        return result

    def save(self) -> None:
        _data_dir.mkdir(parents=True, exist_ok=True)
        with open(_data_dir / f"{self.model_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        ckpt_dir = _data_dir / "checkpoints" / self.model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for ckpt in self.checkpoints:
            torch.save(ckpt, ckpt_dir / f"checkpoint_step{ckpt['step']}_seed{ckpt['seed']}_lr{ckpt['lr']}_bs{ckpt['batch_size']}.pt")


# --- Training ---

def train(model: nn.Module, measure: Measure, epochs: int = 10, batch_size: int = 64,
          lr: float = 1e-3, seed: int = 42) -> Measure:
    torch.manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)

    step = 0
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = masked_loss(logits, y)
            loss.backward()
            optimizer.step()
            measure.record(loss.item(), model, val_loader, step=step, seed=seed, lr=lr, batch_size=batch_size)
            step += 1

    return measure


_seeds = [0,1,2,3,4,5,6,7,8,9]
_lrs = [1e-2, 1e-3, 1e-4]
_bs = [32, 64, 128]

def run(models: list[type[nn.Module]], seeds: list[int] = _seeds, lrs: list[float] = _lrs,
        batch_sizes: list[int] = _bs, epochs: int = 10) -> list[Measure]:
    measures = []
    for model in models:
        measure = Measure(model_name=model.__name__.lower())
        for seed in seeds:
            for lr in lrs:
                for batch_size in batch_sizes:
                    train(model(), measure, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        measure.save()
        measures.append(measure)
    return measures


# --- Helpers ---

def masked_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    targets = y.clone()
    targets[:, :SEQ_LEN] = -100
    return nn.functional.cross_entropy(logits.permute(0, 2, 1), targets, ignore_index=-100)

def evaluate(model: nn.Module, loader: DataLoader) -> float:
    training = model.training
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = masked_loss(logits, y)
            total_loss += loss.item()
    model.train(training)
    return total_loss / len(loader)
