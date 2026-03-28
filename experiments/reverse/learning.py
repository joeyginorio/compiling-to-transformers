from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from experiments.reverse.dataset import generate_datasets, SEQ_LEN

# --- Globals --- 

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# --- Measurements ---

@dataclass
class Measure:
    train_loss: dict[int, float] = field(default_factory=dict)
    val_loss: dict[int, float] = field(default_factory=dict)
    _step: int = field(default=0, repr=False)
    _every: int = field(default=10, repr=False)

    def record(self, loss: float, model: nn.Module, val_loader: DataLoader) -> None:
        self._step += 1
        if self._step % self._every == 0:
            val_loss = evaluate(model, val_loader)
            self.train_loss[self._step] = loss
            self.val_loss[self._step] = val_loss
            print(f"step {self._step:5d}  train {loss:.4f}  val {val_loss:.4f}")


# --- Training ---

def train(model: nn.Module, epochs: int = 20, batch_size: int = 64, lr: float = 1e-3, seed: int = 42, measure_every: int = 10) -> Measure:
    torch.manual_seed(seed)
    datasets = generate_datasets()
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)
    measures = Measure(_every=measure_every)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = masked_loss(logits, y)
            loss.backward()
            optimizer.step()
            measures.record(loss.item(), model, val_loader)

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
