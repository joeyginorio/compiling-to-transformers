import ast
import joblib
import torch
import plotly.express as px  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA as SklearnPCA  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader
from experiments.reverse.models import ModelD, ModelT, ModelI, ModelU
from experiments.reverse.dataset import datasets, decode, encode, SEQ_LEN
from experiments.reverse.learning import evaluate, _seeds, DEVICE

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
_model_cls = {"modeld": ModelD, "modelu": ModelU, "modelt": ModelT, "modeli": ModelI}


def add_test_loss(lr: float, batch_size: int, max_step: int = 800) -> None:
    """Compute test loss from checkpoints and write it into each model's CSV.

    Evaluates every (seed, step) checkpoint at multiples of 50 up to max_step
    for the given lr and batch_size, then writes a test_loss column back to the CSV.
    """
    test_loader = DataLoader(datasets["test"], batch_size=256)
    test_steps = range(0, max_step + 1, 50)

    for model_name in models:
        csv_path = _data_dir / f"{model_name}.csv"
        df = pd.read_csv(csv_path)
        loss_map: dict[tuple[int, int], float] = {}
        for seed in _seeds:
            for step in test_steps:
                ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
                if not ckpt_path.exists():
                    continue
                m = _model_cls[model_name]()
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                m.load_state_dict(ckpt["state_dict"])
                m.to(DEVICE)
                loss_map[(step, seed)] = evaluate(m, test_loader)
        df["test_loss"] = df.apply(
            lambda row: loss_map.get((int(row["step"]), int(row["seed"])), float("nan")),
            axis=1,
        )
        df.to_csv(csv_path, index=False)


def plot_behavioral_dynamics(lr: float | None = None, batch_size: int | None = None, max_step: int = 800, test: bool = False):
    frames: list[pd.DataFrame] = []
    hm_frames: list[pd.DataFrame] = []
    loss_col = "test_loss" if test else "val_loss"

    if test and lr is not None and batch_size is not None:
        sample = pd.read_csv(_data_dir / f"{models[0]}.csv")
        mask = (sample["lr"] == lr) & (sample["batch_size"] == batch_size)
        if "test_loss" not in sample.columns or sample.loc[mask, "test_loss"].isna().all():
            add_test_loss(lr, batch_size, max_step)

    for model_name in models:
        df = pd.read_csv(_data_dir / f"{model_name}.csv")

        # If config given, use that
        if lr is not None and batch_size is not None:
            best_lr, best_bs = lr, batch_size

        # Otherwise find best config
        else:
            last_n = df.groupby(["lr", "batch_size"]).apply(
                lambda g: g.nlargest(100, "step")[loss_col].mean()
            )
            idx = last_n.idxmin()
            assert isinstance(idx, tuple)
            best_lr, best_bs = idx

        run_df = df[(df["lr"] == best_lr) & (df["batch_size"] == best_bs)].copy()
        model_label = model_name[-1].upper()
        run_df["model"] = model_label
        frames.append(run_df[["step", "seed", "train_loss", loss_col, "model"]])  # type: ignore

        # Build heatmap rows for this model using its best config
        run_df["output2"] = run_df["outputs"].apply(lambda s: ast.literal_eval(s)[1])  # type: ignore
        for i, ch in enumerate("abcdefghijklmno"):
            result = run_df.groupby("step")["output2"].apply(  # type: ignore
                lambda g, c=ch, pos=i: (g.str[pos] == c).mean()
            ).reset_index(name="prop_correct")
            result["char"] = ch
            result["model"] = model_label
            hm_frames.append(result)

    data = pd.concat(frames, ignore_index=True)
    hm_data = pd.concat(hm_frames, ignore_index=True)

    fig, (*ax_hms, ax_train, ax_val) = plt.subplots(6, 1, figsize=(8, 14))
    ax_val.sharex(ax_train)
    ax_train.sharey(ax_val)

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
    sns.lineplot(data=data, x="step", y="train_loss", hue="model",
                 palette=palette, hue_order=hue_order, ax=ax_train, errorbar="sd")
    ax_train.set_ylabel("Train Loss")
    ax_train.set_xlabel("")
    ax_train.legend(title="Model")
    ax_train.set_xlim(0, max_step)
    ax_train.tick_params(labelbottom=False)

    # Val/Test loss
    sns.lineplot(data=data, x="step", y=loss_col, hue="model",
                 palette=palette, hue_order=hue_order, ax=ax_val, errorbar="sd")
    ax_val.set_xlabel("Step")
    ax_val.set_ylim(bottom=-0.05)
    ax_val.set_ylabel("Test Loss" if test else "Val Loss")
    ax_val.legend(title="Model")

    fig.suptitle("Behavioral Dynamics", fontsize=18)
    fig.set_size_inches(8, 12)
    fig.tight_layout()
    lr_tag = f"_lr{lr}" if lr is not None else "_lrbest"
    bs_tag = f"_bs{batch_size}" if batch_size is not None else "_bsbest"
    fname = f"behavioral_dynamics{lr_tag}{bs_tag}_maxstep{max_step}.pdf"
    fig.savefig(_data_dir.parent / "plots" / fname)
    plt.show()


# --- PCA ---

def pca(model: nn.Module, step: int, lr: float, batch_size: int, seed: int,
        positions: list[int] | None = None,
        ablate: dict[str, bool] | None = None) -> dict[str, tuple[SklearnPCA, np.ndarray]]:
    model_name = model.__class__.__name__.lower()
    ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(datasets["probe"], batch_size=1024)
    rep_lists: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for x, _ in loader:
            if ablate is not None:
                _, reps = model(x, representations=True, ablate=ablate)
            else:
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

# plot_pca(ModelI(), 2500, .001, 32, 3, [16,17,18], 'r1')
def plot_pca(model: nn.Module, step: int, lr: float, batch_size: int, seed: int,
             positions: list[int], rep: str = "r1",
             ablate: dict[str, bool] | None = None,
             target: bool = True) -> None:
    result = pca(model, step, lr, batch_size, seed, positions=positions, ablate=ablate)
    fitted, matrix = result[rep]

    projected = fitted.transform(matrix)[:, :2]
    n_pos = len(positions)
    N = len(matrix) // n_pos
    pos_array = np.tile(positions, N)

    all_tokens = torch.stack([datasets["probe"][i][0] for i in range(len(datasets["probe"]))])
    target_tokens = all_tokens[:, [p + 1 for p in positions]].numpy().reshape(N * n_pos)
    target_chars = np.array([decode[t] for t in target_tokens])
    input_tokens = all_tokens[:, [p - SEQ_LEN for p in positions]].numpy().reshape(N * n_pos)
    input_chars = np.array([decode[t] for t in input_tokens])

    df = pd.DataFrame(projected, columns=["PC1", "PC2"])  # type: ignore
    df["target"] = target_chars
    df["input"] = input_chars
    df["Position"] = pos_array

    var = fitted.explained_variance_ratio_
    model_key = model.__class__.__name__[-1].upper()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
    title = fig.suptitle(f"Model {model_key}", fontsize=18)
    fig.canvas.draw()
    renderer = getattr(fig.canvas, "renderer", None)
    bb = title.get_window_extent(renderer=renderer)
    bb_fig = bb.transformed(fig.transFigure.inverted())
    fig.add_artist(mlines.Line2D(
        [bb_fig.x0, bb_fig.x1], [bb_fig.y0, bb_fig.y0],
        transform=fig.transFigure,
        color=palette[model_key], linewidth=2
    ))

    # top: colored by target or input token
    hue_col = "target" if target else "input"
    sns.scatterplot(data=df, x="PC1", y="PC2", hue=hue_col,
                    palette="turbo", alpha=0.5, s=10, ax=ax1, legend=False)
    for char, group in df.groupby(hue_col):
        cx, cy = float(group["PC1"].mean()), float(group["PC2"].mean())
        ax1.text(cx, cy, str(char), fontsize=11, fontweight="bold", ha="center", va="center",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.5,
                           alpha=0.6, boxstyle="circle,pad=0.5"))
    ax1.set_aspect("equal")
    ax1.set_xlabel("")
    ax1.set_ylabel(f"PC2 ({var[1]:.1%})")

    # bottom: colored by position
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Position",
                    palette="turbo", alpha=0.5, s=10, ax=ax2)
    ax2.set_aspect("equal")
    ax2.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({var[1]:.1%})")

    fname = f"pca_{model_key}_step{step}_lr{lr}_bs{batch_size}_seed{seed}_{rep}.pdf"
    plt.tight_layout()
    fig.savefig(_data_dir.parent / "plots" / fname)
    plt.show()


def plot_pca_3d(model: nn.Module, step: int, lr: float, batch_size: int, seed: int,
                positions: list[int], rep: str = "r1",
                ablate: dict[str, bool] | None = None,
                target: bool = True) -> None:
    result = pca(model, step, lr, batch_size, seed, positions=positions, ablate=ablate)
    fitted, matrix = result[rep]

    projected = fitted.transform(matrix)[:, :3]
    n_pos = len(positions)
    N = len(matrix) // n_pos
    pos_array = np.tile(positions, N)

    all_tokens = torch.stack([datasets["probe"][i][0] for i in range(len(datasets["probe"]))])
    target_tokens = all_tokens[:, [p + 1 for p in positions]].numpy().reshape(N * n_pos)
    target_chars = np.array([decode[t] for t in target_tokens])
    input_tokens = all_tokens[:, [p - SEQ_LEN for p in positions]].numpy().reshape(N * n_pos)
    input_chars = np.array([decode[t] for t in input_tokens])

    var = fitted.explained_variance_ratio_
    model_key = model.__class__.__name__[-1].upper()
    hue_col = "target" if target else "input"

    df = pd.DataFrame(projected, columns=["PC1", "PC2", "PC3"])  # type: ignore
    df["target"] = target_chars
    df["input"] = input_chars
    df["Position"] = pos_array.astype(str)

    axis_labels = {
        "PC1": f"PC1 ({var[0]:.1%})",
        "PC2": f"PC2 ({var[1]:.1%})",
        "PC3": f"PC3 ({var[2]:.1%})",
    }

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                        subplot_titles=[hue_col.capitalize(), "Position"])

    for trace in px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color=hue_col, opacity=0.5).data:
        fig.add_trace(trace, row=1, col=1)

    for trace in px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color="Position", opacity=0.5).data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(title=f"Model {model_key} — {rep}",
                      scene=dict(xaxis_title=axis_labels["PC1"], yaxis_title=axis_labels["PC2"], zaxis_title=axis_labels["PC3"]),
                      scene2=dict(xaxis_title=axis_labels["PC1"], yaxis_title=axis_labels["PC2"], zaxis_title=axis_labels["PC3"]))
    fig.show()


def plot_pca_all(models_list: list[nn.Module], step: int, lr: float, batch_size: int, seed: int,
                 positions: list[int], rep: str = "r1", 
                 ablate: dict[str, bool] | None = None,
                 target: bool = True) -> None:
    n = len(models_list)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=True)

    for col, model in enumerate(models_list):
        result = pca(model, step, lr, batch_size, seed, positions=positions, ablate=ablate)
        fitted, matrix = result[rep]

        projected = fitted.transform(matrix)[:, :2]
        n_pos = len(positions)
        N = len(matrix) // n_pos
        pos_array = np.tile(positions, N)

        all_tokens = torch.stack([datasets["probe"][i][0] for i in range(len(datasets["probe"]))])
        target_tokens = all_tokens[:, [p + 1 for p in positions]].numpy().reshape(N * n_pos)
        target_chars = np.array([decode[t] for t in target_tokens])
        
        # In case you want to probe input token
        input_tokens = all_tokens[:, [p - SEQ_LEN for p in positions]].numpy().reshape(N * n_pos)
        input_chars = np.array([decode[t] for t in input_tokens])

        df = pd.DataFrame(projected, columns=["PC1", "PC2"])  # type: ignore
        df["target"] = target_chars
        df["input"] = input_chars
        df["Position"] = pos_array

        var = fitted.explained_variance_ratio_
        model_key = model.__class__.__name__[-1].upper()

        ax1, ax2 = axes[0, col], axes[1, col]
        ax1.set_title(f"Model {model_key}", fontsize=18, color=palette[model_key], fontweight="bold", pad=12)

        # top: colored by target or input token
        hue_col = "target" if target else "input"
        sns.scatterplot(data=df, x="PC1", y="PC2", hue=hue_col,
                        palette="turbo", alpha=0.5, s=10, ax=ax1, legend=False)
        for char, group in df.groupby(hue_col):
            cx, cy = float(group["PC1"].mean()), float(group["PC2"].mean())
            ax1.text(cx, cy, str(char), fontsize=11, fontweight="bold", ha="center", va="center",
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.5,
                               alpha=0.6, boxstyle="circle,pad=0.5"))
        ax1.set_anchor("N")
        ax1.set_xlabel("")
        ax1.set_ylabel(f"PC2 ({var[1]:.1%})")
        if col == 0:
            cmap = plt.cm.get_cmap("turbo")
            n_swatches = 6
            handles = [mpatches.Patch(color=cmap(i / (n_swatches - 1)), alpha=0.5) for i in range(n_swatches)]
            ax1.legend(handles=handles, labels=[""] * n_swatches, title="Target Token",
                       loc="upper left", ncol=n_swatches, handlelength=1,
                       handletextpad=0, columnspacing=0.2)

        # bottom: colored by position
        sns.scatterplot(data=df, x="PC1", y="PC2", hue="Position",
                        palette="turbo", alpha=0.5, s=10, ax=ax2)
        ax2.set_anchor("N")
        if col == 0:
            ax2.legend(loc="upper left", title="Position")
        else:
            ax2.legend_.remove() if ax2.legend_ else None
        ax2.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax2.set_ylabel(f"PC2 ({var[1]:.1%})")

    fname = f"pca_all_step{step}_lr{lr}_bs{batch_size}_seed{seed}_{rep}.pdf"
    fig.suptitle(f"PCA at {rep.upper()}", fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(_data_dir.parent / "plots" / fname)
    plt.show()


# --- Ablations ---

def plot_ablations(step: int, lr: float, batch_size: int, test: bool = False) -> None:
    model_configs: list[tuple[str, nn.Module, list[tuple[str, dict[str, bool]]]]] = [
        ("D", ModelD(), [
            ("_",   {'h1': False, 'rev_out': False}),
            ("A",   {'h1': True,  'rev_out': False}),
            ("R",   {'h1': False, 'rev_out': True}),
            ("A+R", {'h1': True,  'rev_out': True}),
        ]),
        ("U", ModelU(), [
            ("_",   {'h1': False, 'rev_out': False}),
            ("A",   {'h1': True,  'rev_out': False}),
            ("R",   {'h1': False, 'rev_out': True}),
            ("A+R", {'h1': True,  'rev_out': True}),
        ]),
        ("T", ModelT(), [
            ("_",   {'h1': False, 'rev_out': False}),
            ("A",   {'h1': True,  'rev_out': False}),
            ("R",   {'h1': False, 'rev_out': True}),
            ("A+R", {'h1': True,  'rev_out': True}),
        ]),
        ("I", ModelI(), [
            ("_",   {'h1': False}),
            ("A",   {'h1': True}),
        ]),
    ]
    all_conditions = ["_", "A", "R", "A+R"]

    val_loader = DataLoader(datasets["test" if test else "val"], batch_size=256)
    rows: list[dict] = []

    for model_label, model, conditions in model_configs:
        model_name = f"model{model_label.lower()}"
        for seed in _seeds:
            ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])

            for label, ablate in conditions:
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        logits = model(x, ablate=ablate)
                        preds = logits[:, 15:, :].argmax(dim=-1)
                        targets = y[:, 15:]
                        correct += (preds == targets).sum().item()
                        total += targets.numel()
                rows.append({"condition": label, "model": model_label, "accuracy": correct / total})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="condition", y="accuracy", hue="model",
                order=all_conditions, hue_order=hue_order,
                palette=palette, errorbar="sd", capsize=0.1, ax=ax, width=0.6,
                err_kws={"color": "black", "linewidth": 1.5})
    ax.axhline(1 / 26, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("Ablation")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    fig.suptitle("Ablations on Attention (A) and Reverse (R)", fontsize=18)
    fig.tight_layout()
    fname = f"ablations_step{step}_lr{lr}_bs{batch_size}.pdf"
    fig.savefig(_data_dir.parent / "plots" / fname)
    plt.show()


# --- Probes ---

def train_probes(model: nn.Module, lr: float, batch_size: int, steps: list[int] = [0, 250, 2500], rep: str = "rev_in") -> None:
    """Train and save a linear probe for each (step, seed) checkpoint.
    Saves each fitted probe as a .joblib file and writes a CSV of probe/task accuracies."""
    model_name = model.__class__.__name__.lower()
    probes_dir = _data_dir / "probes"
    rows = []

    for step in steps:
        for seed in _seeds:
            ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            # Extract representations
            probe_loader = DataLoader(datasets["probe"], batch_size=1024)
            rep_list, toks_list = [], []
            with torch.no_grad():
                for x, _ in probe_loader:
                    _, reps = model(x, representations=True)
                    rep_list.append(reps[rep][:, SEQ_LEN:, :])
                    toks_list.append(x[:, :SEQ_LEN])

            rep_tensor = torch.cat(rep_list, dim=0)  # [N, SEQ_LEN, d]
            tokens = torch.cat(toks_list, dim=0)     # [N, SEQ_LEN]
            N = rep_tensor.shape[0]
            X = rep_tensor.reshape(N * SEQ_LEN, -1).numpy()
            y_probe = tokens.reshape(N * SEQ_LEN).numpy()

            X_train, X_test, y_train, y_test = train_test_split(X, y_probe, test_size=0.2, random_state=42)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            probe_acc = float(clf.score(X_test, y_test))

            joblib.dump(clf, probes_dir / f"probe_{model_name}_{rep}_step{step}_seed{seed}_lr{lr}_bs{batch_size}.joblib")

            # Task accuracy
            val_loader = DataLoader(datasets["val"], batch_size=1024)
            correct = total = 0
            with torch.no_grad():
                for x, y_task in val_loader:
                    logits = model(x)
                    preds = logits[:, SEQ_LEN:, :].argmax(dim=-1)
                    targets = y_task[:, SEQ_LEN:]
                    correct += int((preds == targets).sum().item())
                    total += targets.numel()
            task_acc = correct / total

            rows.append({"step": step, "seed": seed, "probe_accuracy": probe_acc, "task_accuracy": task_acc})

    pd.DataFrame(rows).to_csv(probes_dir / f"probe_{model_name}_{rep}_lr{lr}_bs{batch_size}.csv", index=False)


def add_test_task_accuracy(lr: float, batch_size: int, rep: str = "rev_in") -> None:
    """Compute test task accuracy from saved checkpoints and write it into the probe CSV for models D, T, U."""
    test_loader = DataLoader(datasets["test"], batch_size=1024)

    for model_name, model_cls in [("modeld", ModelD), ("modelt", ModelT), ("modelu", ModelU)]:
        csv_path = _data_dir / "probes" / f"probe_{model_name}_{rep}_lr{lr}_bs{batch_size}.csv"
        df = pd.read_csv(csv_path)
        steps: list[int] = [int(s) for s in df["step"].unique()]
        m = model_cls()
        acc_map: dict[tuple[int, int], float] = {}

        for step in steps:
            for seed in _seeds:
                ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                m.load_state_dict(ckpt["state_dict"])
                m.eval()
                correct = total = 0
                with torch.no_grad():
                    for x, y_task in test_loader:
                        logits = m(x)
                        preds = logits[:, SEQ_LEN:, :].argmax(dim=-1)
                        targets = y_task[:, SEQ_LEN:]
                        correct += int((preds == targets).sum().item())
                        total += targets.numel()
                acc_map[(step, seed)] = correct / total

        df["test_task_accuracy"] = df.apply(
            lambda row: acc_map.get((int(row["step"]), int(row["seed"])), float("nan")),
            axis=1,
        )
        df.to_csv(csv_path, index=False)


def plot_probe(model: nn.Module, lr: float, batch_size: int, rep: str = "rev_in", test: bool = False) -> None:
    """Plot task accuracy and probe accuracy from saved probe results CSV."""
    model_name = model.__class__.__name__.lower()
    csv_path = _data_dir / "probes" / f"probe_{model_name}_{rep}_lr{lr}_bs{batch_size}.csv"
    results = pd.read_csv(csv_path)
    if test and ("test_task_accuracy" not in results.columns or bool(results["test_task_accuracy"].isna().all())):
        add_test_task_accuracy(lr, batch_size, rep=rep)
        results = pd.read_csv(csv_path)

    model_key = model.__class__.__name__[-1].upper()
    task_col = "test_task_accuracy" if test else "task_accuracy"

    fig, ax = plt.subplots(figsize=(7, 4))
    title = fig.suptitle(f"Model {model_key}", fontsize=18)
    fig.canvas.draw()
    renderer = getattr(fig.canvas, "renderer", None)
    bb = title.get_window_extent(renderer=renderer)
    bb_fig = bb.transformed(fig.transFigure.inverted())
    fig.add_artist(mlines.Line2D(
        [bb_fig.x0, bb_fig.x1], [bb_fig.y0, bb_fig.y0],
        transform=fig.transFigure,
        color=palette[model_key], linewidth=2
    ))

    sns.lineplot(data=results, x="step", y=task_col, ax=ax, color=palette[model_key],
                 errorbar="sd", label="Task Accuracy", marker="s", linestyle=":")
    sns.lineplot(data=results, x="step", y="probe_accuracy", ax=ax, color=palette[model_key],
                 errorbar="sd", label="Probe Accuracy", marker="o", linestyle="-")
    ax.axhline(1 / 26, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_data_dir.parent / "plots" / f"probe_{model_name}_{rep}_lr{lr}_bs{batch_size}.pdf")
    plt.show()


def plot_probe_all(models_list: list[nn.Module], lr: float, batch_size: int, rep: str = "rev_in", test: bool = False) -> None:
    n = len(models_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    task_col = "test_task_accuracy" if test else "task_accuracy"

    if test:
        first_name = models_list[0].__class__.__name__.lower()
        first_csv = pd.read_csv(_data_dir / "probes" / f"probe_{first_name}_{rep}_lr{lr}_bs{batch_size}.csv")
        if "test_task_accuracy" not in first_csv.columns or bool(first_csv["test_task_accuracy"].isna().all()):
            add_test_task_accuracy(lr, batch_size, rep=rep)

    for col, model in enumerate(models_list):
        model_name = model.__class__.__name__.lower()
        model_key = model.__class__.__name__[-1].upper()
        csv_path = _data_dir / "probes" / f"probe_{model_name}_{rep}_lr{lr}_bs{batch_size}.csv"
        results = pd.read_csv(csv_path)

        ax = axes[col]
        ax.set_title(f"Model {model_key}", fontsize=18, color=palette[model_key], fontweight="bold", pad=12)

        sns.lineplot(data=results, x="step", y=task_col, ax=ax, color=palette[model_key],
                     errorbar="sd", label="Task Accuracy", marker="s", linestyle=":")
        sns.lineplot(data=results, x="step", y="probe_accuracy", ax=ax, color=palette[model_key],
                     errorbar="sd", label="Probe Accuracy", marker="o", linestyle="-")
        ax.axhline(1 / 26, color="black", linestyle="--", linewidth=1, label="Chance")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy" if col == 0 else "")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 600)
        if col == 0:
            legend_handles = [
                mlines.Line2D([], [], color="black", marker="s", linestyle=":", label="Task Accuracy"),
                mlines.Line2D([], [], color="black", marker="o", linestyle="-", label="Probe Accuracy"),
                mlines.Line2D([], [], color="black", linestyle="--", linewidth=1, label="Chance"),
            ]
            ax.legend(handles=legend_handles)
        else:
            ax.legend_.remove() if ax.legend_ else None

    fig.suptitle("Do correct embeddings of token identity drive behavior?", fontsize=18)
    fig.tight_layout()
    fig.savefig(_data_dir.parent / "plots" / f"probe_all_{rep}_lr{lr}_bs{batch_size}.pdf")
    plt.show()


# --- Steering ---

def plot_steering(step: int, lr: float, batch_size: int, n_targets: int = 50, k: int = 1, test: bool = False) -> None:
    """Plot steerability vs alpha for Models D, U, T.

    For each alpha, steers all rev_in positions simultaneously toward a random
    input string s and measures the fraction of output positions where the target
    token (s[::-1][i]) appears in the top-k logits. k=1 is exact argmax accuracy.
    Averaged over n_targets random strings and all seeds.
    """
    import random as _random

    model_configs = [("D", ModelD()), ("U", ModelU()), ("T", ModelT())]
    alphas = [0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    letters = [decode[i] for i in range(26)]

    rng = _random.Random(42)
    inputs = [''.join(rng.choices(letters, k=SEQ_LEN)) for _ in range(n_targets)]

    split = "test" if test else "val"
    x_test = torch.stack([datasets[split][i][0] for i in range(len(datasets[split]))])

    rows: list[dict] = []

    for model_label, model in model_configs:
        model_name = f"model{model_label.lower()}"

        for seed in _seeds:
            ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            probe_path = _data_dir / "probes" / f"probe_{model_name}_rev_in_step{step}_seed{seed}_lr{lr}_bs{batch_size}.joblib"
            clf = joblib.load(probe_path)
            W = torch.tensor(clf.coef_, dtype=torch.float32)  # [n_classes, d_head]
            class_to_idx = {int(c): i for i, c in enumerate(clf.classes_)}

            with torch.no_grad():
                for alpha in alphas:
                    correct = total = 0
                    for s in inputs:
                        steer = {p: alpha * W[class_to_idx[encode[s[p]]]]
                                 for p in range(SEQ_LEN)}
                        logits = model(x_test, steer=steer)
                        topk = logits[:, SEQ_LEN:2 * SEQ_LEN, :].topk(k, dim=-1).indices  # [B, SEQ_LEN, k]
                        target_tokens = torch.tensor([encode[c] for c in s[::-1]])          # [SEQ_LEN]
                        correct += (topk == target_tokens.unsqueeze(0).unsqueeze(-1)).any(dim=-1).sum().item()
                        total += topk.shape[0] * topk.shape[1]
                    rows.append({"model": model_label, "alpha": alpha, "seed": seed,
                                 "steerability": correct / total})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df, x="alpha", y="steerability", hue="model",
                 palette={m: palette[m] for m in ["D", "U", "T"]},
                 hue_order=["D", "U", "T"],
                 errorbar="sd", marker="o", ax=ax)

    ax.set_xlabel("α")
    ax.set_ylabel("Mean Steerability (Top-K)")
    ax.legend(title="Model")
    fig.suptitle("Steerability: steering rev_in to reverse random inputs", fontsize=18)
    fig.tight_layout()
    fig.savefig(_data_dir.parent / "plots" / f"steering_step{step}_lr{lr}_bs{batch_size}_k{k}.pdf")
    plt.show()


def plot_steering_all(step: int, lr: float, batch_size: int, n_targets: int = 50, test: bool = False) -> None:
    """Side-by-side steerability plots for k=1, 2, 3.

    Runs forward passes once per (model, seed, alpha, input string) and computes
    top-k accuracy for all three k values simultaneously.
    """
    import random as _random

    model_configs = [("D", ModelD()), ("U", ModelU()), ("T", ModelT())]
    alphas = [0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    ks = [1, 2, 3]
    letters = [decode[i] for i in range(26)]

    rng = _random.Random(42)
    inputs = [''.join(rng.choices(letters, k=SEQ_LEN)) for _ in range(n_targets)]

    split = "test" if test else "val"
    x_test = torch.stack([datasets[split][i][0] for i in range(len(datasets[split]))])

    rows: list[dict] = []

    for model_label, model in model_configs:
        model_name = f"model{model_label.lower()}"

        for seed in _seeds:
            ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            probe_path = _data_dir / "probes" / f"probe_{model_name}_rev_in_step{step}_seed{seed}_lr{lr}_bs{batch_size}.joblib"
            clf = joblib.load(probe_path)
            W = torch.tensor(clf.coef_, dtype=torch.float32)
            class_to_idx = {int(c): i for i, c in enumerate(clf.classes_)}

            with torch.no_grad():
                for alpha in alphas:
                    counts = {k: 0 for k in ks}
                    total = 0
                    for s in inputs:
                        steer = {p: alpha * W[class_to_idx[encode[s[p]]]]
                                 for p in range(SEQ_LEN)}
                        logits = model(x_test, steer=steer)
                        top3 = logits[:, SEQ_LEN:2 * SEQ_LEN, :].topk(3, dim=-1).indices  # [B, SEQ_LEN, 3]
                        target_tokens = torch.tensor([encode[c] for c in s[::-1]])          # [SEQ_LEN]
                        target_exp = target_tokens.unsqueeze(0).unsqueeze(-1)               # [1, SEQ_LEN, 1]
                        matches = (top3 == target_exp)                                      # [B, SEQ_LEN, 3]
                        for k in ks:
                            counts[k] += matches[:, :, :k].any(dim=-1).sum().item()
                        total += top3.shape[0] * top3.shape[1]
                    for k in ks:
                        rows.append({"model": model_label, "alpha": alpha, "seed": seed,
                                     "k": k, "steerability": counts[k] / total})

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for col, k in enumerate(ks):
        ax = axes[col]
        sns.lineplot(data=df[df["k"] == k].copy(), x="alpha", y="steerability", hue="model",  # type: ignore[arg-type]
                     palette={m: palette[m] for m in ["D", "U", "T"]},
                     hue_order=["D", "U", "T"],
                     errorbar="sd", marker="o", ax=ax)
        ax.set_title(f"K={k}", fontsize=14)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Mean Steerability (Top-K)" if col == 0 else "")
        if col == 0:
            ax.legend(title="Model")
        else:
            ax.legend_.remove()

    fig.suptitle("Can these embeddings be steered to change behavior?", fontsize=18)
    fig.tight_layout()
    fig.savefig(_data_dir.parent / "plots" / f"steering_all_step{step}_lr{lr}_bs{batch_size}.pdf")
    plt.show()


# --- Helpers ---

# For model, step, and position--fraction of seeds where output2[i] == target[i].
def prop_correct(lr: float, batch_size: int) -> pd.DataFrame:
    rows = []
    for model_name in models:
        df = pd.read_csv(_data_dir / f"{model_name}.csv")
        df = df[(df["lr"] == lr) & (df["batch_size"] == batch_size)].copy()
        df["output2"] = df["outputs"].apply(lambda s: ast.literal_eval(s)[1])  # type: ignore
        for i, ch in enumerate("abcdefghijklmno"):
            result = df.groupby("step")["output2"].apply(  # type: ignore
                lambda g, c=ch, idx=i: (g.str[idx] == c).mean()
            ).reset_index(name="prop_correct")
            result["char"] = ch
            result["model"] = model_name[-1].upper()
            rows.append(result)
    return pd.concat(rows, ignore_index=True)
