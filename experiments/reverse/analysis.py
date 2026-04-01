import ast
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA as SklearnPCA
from torch.utils.data import DataLoader
from experiments.reverse.models import ModelD, ModelT, ModelI
from experiments.reverse.dataset import datasets

# -- Globals ---

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

bmidnight = (0.05, 0.33, 0.62)
bcayenne = (0.68, 0.08, 0.14)
purple = (0.82, 0.42, 0.05)
green  = (0.02, 0.48, 0.50)

palette = {
    "D": bmidnight,
    "U": purple,
    "T": green,
    "I": bcayenne,
}

hue_order = ["D", "U", "T", "I"]

sns.set_theme(style="white",
              font="Futura",
              rc={
                "font.weight": "bold",
                "axes.labelsize": 15,
                "xtick.labelsize": 12,
                "ytick.labelsize": 10})

_data_dir = Path("experiments/reverse/data/learning")
models = ["modeld", "modelu", "modelt", "modeli"]


# --- Behavioral Dynamics ---

def behavioral_dynamics(lr: float | None = None, batch_size: int | None = None, max_step: int = 800):
    frames: list[pd.DataFrame] = []
    hm_frames: list[pd.DataFrame] = []

    for model_name in models:
        df = pd.read_csv(_data_dir / f"{model_name}.csv")

        # If config given, use that
        if lr is not None and batch_size is not None:
            best_lr, best_bs = lr, batch_size

        # Otherwise find best config
        else:
            last_n = df.groupby(["lr", "batch_size"]).apply(
                lambda g: g.nlargest(100, "step")["val_loss"].mean()
            )
            idx = last_n.idxmin()
            assert isinstance(idx, tuple)
            best_lr, best_bs = idx

        run_df = df[(df["lr"] == best_lr) & (df["batch_size"] == best_bs)].copy()
        model_label = model_name[-1].upper()
        run_df["model"] = model_label
        frames.append(run_df[["step", "seed", "train_loss", "val_loss", "model"]])

        # Build heatmap rows for this model using its best config
        run_df["output2"] = run_df["outputs"].apply(lambda s: ast.literal_eval(s)[1])
        for i, ch in enumerate("abcdefghijklmno"):
            result = run_df.groupby("step")["output2"].apply(
                lambda g, c=ch, pos=i: (g.str[pos] == c).mean()
            ).reset_index(name="prop_correct")
            result["char"] = ch
            result["model"] = model_label
            hm_frames.append(result)

    data = pd.concat(frames, ignore_index=True)
    hm_data = pd.concat(hm_frames, ignore_index=True)

    fig, (*ax_hms, ax_train, ax_val) = plt.subplots(6, 1, figsize=(8, 14))
    ax_val.sharex(ax_train)

    # Model Heatmaps
    for ax_hm, model_label in zip(ax_hms, hue_order):
        hm = hm_data[hm_data["model"] == model_label].pivot(index="char", columns="step", values="prop_correct")
        hm = hm.loc[:, hm.columns.astype(int) <= max_step]
        cmap = sns.light_palette(palette[model_label], as_cmap=True)
        sns.heatmap(hm, ax=ax_hm, vmin=0, vmax=1, cmap=cmap, cbar=False,
                    xticklabels=50, yticklabels=True)
        cax = ax_hm.inset_axes([1.01, 0.1, 0.02, 0.8])
        fig.colorbar(ax_hm.collections[0], cax=cax)
        ax_hm.set_xlabel("")
        ax_hm.set_ylabel(model_label)
        ax_hm.tick_params(labelbottom=False)

    # Train loss
    sns.lineplot(data=data, x="step", y="train_loss", hue="model", units="seed",
                 estimator=None, palette=palette, hue_order=hue_order, ax=ax_train)
    ax_train.set_ylabel("Train Loss")
    ax_train.set_xlabel("")
    ax_train.legend(title="Model")
    ax_train.set_xlim(0, max_step)
    ax_train.tick_params(labelbottom=False)

    # Val loss
    sns.lineplot(data=data, x="step", y="val_loss", hue="model", units="seed",
                 estimator=None, palette=palette, hue_order=hue_order, ax=ax_val)
    ax_val.set_xlabel("Step")
    ax_val.set_ylabel("Val Loss")
    ax_val.legend(title="Model")

    fig.suptitle("Behavioral Dynamics", fontsize=18)
    fig.set_size_inches(8, 12)
    fig.tight_layout()
    fig.savefig(_data_dir.parent / "plots" / "behavioral_dynamics.pdf")
    plt.show()


# --- PCA ---
def pca(model: nn.Module, step: int, lr: float, batch_size: int, seed: int,
        positions: list[int] | None = None) -> dict[str, tuple[SklearnPCA, np.ndarray]]:
    model_name = model.__class__.__name__.lower()
    ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(datasets["probe"], batch_size=1024)
    rep_lists: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for x, _ in loader:
            _, reps = model(x, representations=True)
            for key, tensor in reps.items():
                rep_lists.setdefault(key, []).append(tensor)

    result = {}
    for key, tensors in rep_lists.items():
        r = torch.cat(tensors, dim=0)
        if positions is not None:
            r = r[:, positions, :]
        N, seq_len, d_model = r.shape
        matrix = r.reshape(N * seq_len, d_model).numpy()
        fitted = SklearnPCA()
        fitted.fit(matrix)
        result[key] = (fitted, matrix)
    return result


# --- Probes --- 

# --- Helpers ---

# For model, step, and position--fraction of seeds where output2[i] == target[i].
def prop_correct(lr: float, batch_size: int) -> pd.DataFrame:
    rows = []
    for model_name in models:
        df = pd.read_csv(_data_dir / f"{model_name}.csv")
        df = df[(df["lr"] == lr) & (df["batch_size"] == batch_size)].copy()
        df["output2"] = df["outputs"].apply(lambda s: ast.literal_eval(s)[1])
        for i, ch in enumerate("abcdefghijklmno"):
            result = df.groupby("step")["output2"].apply(
                lambda g, c=ch, idx=i: (g.str[idx] == c).mean()
            ).reset_index(name="prop_correct")
            result["char"] = ch
            result["model"] = model_name[-1].upper()
            rows.append(result)
    return pd.concat(rows, ignore_index=True)
