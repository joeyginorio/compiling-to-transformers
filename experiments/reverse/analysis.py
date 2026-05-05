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
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA as SklearnPCA  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import balanced_accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader
from experiments.reverse.models import ModelF, ModelT, ModelI, ModelU
from experiments.reverse.dataset import datasets, decode, encode, SEQ_LEN, VOWELS, ReverseDataset
from experiments.reverse.learning import evaluate, _seeds, DEVICE

# -- Globals ---
_seeds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

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
    "F": bmidnight,
    "U": purple,
    "T": green,
    "I": bcayenne,
}

hue_order = ["F", "U", "T", "I"]

sns.set_theme(style="white",
              font="Futura",
              rc={
                "font.weight": "bold",
                "axes.labelsize": 15,
                "xtick.labelsize": 12,
                "ytick.labelsize": 10})

_data_dir = Path("experiments/reverse/data/learning")
models = ["modelf", "modelu", "modelt", "modeli"]



# --- Behavioral Dynamics ---
_model_cls = {"modelf": ModelF, "modelu": ModelU, "modelt": ModelT, "modeli": ModelI}


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
        m = _model_cls[model_name]().to(DEVICE)
        loss_map: dict[tuple[int, int], float] = {}
        for seed in _seeds:
            for step in test_steps:
                ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                m.load_state_dict(ckpt["state_dict"])
                loss_map[(step, seed)] = evaluate(m, test_loader)
        df["test_loss"] = df.apply(
            lambda row: loss_map.get((int(row["step"]), int(row["seed"])), float("nan")),
            axis=1,
        )
        df.to_csv(csv_path, index=False)


def plot_behavioral_dynamics(lr: float | None = None, batch_size: int | None = None, max_step: int = 800, test: bool = False):
    plt.rcParams.update({
        'font.size': 12,
        'figure.titlesize': 34,
        'axes.labelsize': 34,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
    })
    frames: list[pd.DataFrame] = []
    hm_frames: list[pd.DataFrame] = []
    loss_col = "test_loss" if test else "val_loss"

    if test and lr is not None and batch_size is not None:
        for mn in models:
            s = pd.read_csv(_data_dir / f"{mn}.csv")
            mask = (s["lr"] == lr) & (s["batch_size"] == batch_size)
            if "test_loss" not in s.columns or s.loc[mask, "test_loss"].isna().all():
                add_test_loss(lr, batch_size, max_step)
                break

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

    fig, (*ax_hms, ax_train, ax_val) = plt.subplots(6, 1, figsize=(8, 20))
    ax_val.sharex(ax_train)
    ax_train.sharey(ax_val)

    # Model Heatmaps
    for ax_hm, model_label in zip(ax_hms, hue_order):
        hm = hm_data[hm_data["model"] == model_label].pivot(index="char", columns="step", values="prop_correct")
        hm = hm.loc[:, hm.columns.astype(int) <= max_step]
        cmap = sns.light_palette(palette[model_label], as_cmap=True)
        sns.heatmap(hm, ax=ax_hm, vmin=0, vmax=1, cmap=cmap, cbar=False,
                    xticklabels=50, yticklabels=False)
        cax = ax_hm.inset_axes([1.01, 0.1, 0.02, 0.8])
        fig.colorbar(ax_hm.collections[0], cax=cax)
        ax_hm.set_xlabel("")
        ax_hm.set_ylabel(f"{model_label}\no–a", color=palette[model_label], fontweight="bold")
        ax_hm.tick_params(labelbottom=False)

    linestyles = {"F": (1, 0), "U": (5, 2), "T": (1, 2), "I": (5, 2, 1, 2)}

    # Train loss
    sns.lineplot(data=data, x="step", y="train_loss", hue="model", style="model",
                 palette=palette, hue_order=hue_order, style_order=hue_order,
                 dashes=linestyles, ax=ax_train,
                 estimator="median", errorbar=("pi", 50), linewidth=3.5)
    ax_train.set_ylabel("Train")
    ax_train.set_xlabel("")
    ax_train.legend_.remove() if ax_train.legend_ else None
    ax_train.set_xlim(0, max_step)
    ax_train.tick_params(labelbottom=False)

    # Val/Test loss
    sns.lineplot(data=data, x="step", y=loss_col, hue="model", style="model",
                 palette=palette, hue_order=hue_order, style_order=hue_order,
                 dashes=linestyles, ax=ax_val, errorbar="sd", linewidth=3.5)
    ax_val.set_xlabel("Step", labelpad=20)
    ax_val.set_ylim(-0.05, 4)
    ax_val.set_ylabel("Test" if test else "Val")
    ax_val.legend_.remove() if ax_val.legend_ else None

    fig.set_size_inches(8, 12)
    fig.tight_layout()
    handles, labels = ax_val.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.92, 0.18),
               bbox_transform=fig.transFigure, frameon=True)
    lr_tag = f"_lr{lr}" if lr is not None else "_lrbest"
    bs_tag = f"_bs{batch_size}" if batch_size is not None else "_bsbest"
    fname = f"behavioral_dynamics{lr_tag}{bs_tag}_maxstep{max_step}.pdf"
    fig.savefig(_data_dir.parent / "plots" / fname, bbox_inches='tight', pad_inches=0)
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
             target: bool = True,
             show_chars: list[str] | None = ["a", "b", "c", "d", "e"]) -> None:
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

    vowel_ids = [encode[c] for c in "aeiou"]
    is_vowel = np.isin(all_tokens[:, 0].numpy(), vowel_ids)
    branch = np.repeat(np.where(is_vowel, "reverse", "identity"), n_pos)

    df = pd.DataFrame(projected, columns=["PC1", "PC2"])  # type: ignore
    df["target"] = target_chars
    df["input"] = input_chars
    df["Position"] = pos_array
    df["branch"] = branch

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
    plot_df = df if show_chars is None else df.loc[df[hue_col].isin(show_chars)]
    id_df = plot_df[plot_df["branch"] == "identity"]
    rev_df = plot_df[plot_df["branch"] == "reverse"]
    char_order = sorted(plot_df[hue_col].unique())
    pos_order = sorted(plot_df["Position"].unique())
    sns.scatterplot(data=id_df, x="PC1", y="PC2", hue=hue_col, hue_order=char_order,  # type: ignore[arg-type]
                    palette="turbo", alpha=0.5, s=20, marker="o",
                    edgecolor="none", ax=ax1, legend=False)
    sns.scatterplot(data=rev_df, x="PC1", y="PC2", hue=hue_col, hue_order=char_order,  # type: ignore[arg-type]
                    palette="turbo", alpha=0.5, s=70, marker="X",
                    edgecolor="black", linewidth=0.3, ax=ax1, legend=False)
    for key, group in plot_df.groupby([hue_col, "branch"]):
        char = key[0]  # type: ignore[index]
        cx, cy = float(group["PC1"].mean()), float(group["PC2"].mean())
        ax1.text(cx, cy, str(char), fontsize=11, fontweight="bold", ha="center", va="center",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.5,
                           alpha=0.6, boxstyle="circle,pad=0.5"))
    ax1.set_aspect("equal")
    ax1.set_xlabel("")
    ax1.set_ylabel(f"PC2 ({var[1]:.1%})")

    # bottom: colored by position
    sns.scatterplot(data=id_df, x="PC1", y="PC2", hue="Position", hue_order=pos_order,  # type: ignore[arg-type]
                    palette="turbo", alpha=0.5, s=20, marker="o",
                    edgecolor="none", ax=ax2, legend=False)
    sns.scatterplot(data=rev_df, x="PC1", y="PC2", hue="Position", hue_order=pos_order,  # type: ignore[arg-type]
                    palette="turbo", alpha=0.5, s=70, marker="X",
                    edgecolor="black", linewidth=0.3, ax=ax2)
    ax2.set_aspect("equal")
    ax2.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({var[1]:.1%})")

    fname = f"pca_{model_key}_step{step}_lr{lr}_bs{batch_size}_seed{seed}_{rep}.pdf"
    plt.tight_layout()
    fig.savefig(_data_dir.parent / "plots" / fname, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_pca_3d(model: nn.Module, step: int, lr: float, batch_size: int, seed: int,
                positions: list[int], rep: str = "r1",
                ablate: dict[str, bool] | None = None,
                target: bool = True,
                show_chars: list[str] | None = ["a", "b", "c", "d", "e"]) -> None:
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

    vowel_ids = [encode[c] for c in "aeiou"]
    is_vowel = np.isin(all_tokens[:, 0].numpy(), vowel_ids)
    branch = np.repeat(np.where(is_vowel, "reverse", "identity"), n_pos)

    var = fitted.explained_variance_ratio_
    model_key = model.__class__.__name__[-1].upper()
    hue_col = "target" if target else "input"

    df = pd.DataFrame(projected, columns=["PC1", "PC2", "PC3"])  # type: ignore
    df["target"] = target_chars
    df["input"] = input_chars
    df["Position"] = pos_array.astype(str)
    df["branch"] = branch

    axis_labels = {
        "PC1": f"PC1 ({var[0]:.1%})",
        "PC2": f"PC2 ({var[1]:.1%})",
        "PC3": f"PC3 ({var[2]:.1%})",
    }

    symbol_map = {"identity": "circle", "reverse": "diamond"}

    plot_df = df if show_chars is None else df.loc[df[hue_col].isin(show_chars)]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                        subplot_titles=[hue_col.capitalize(), "Position"])

    for trace in px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color=hue_col,
                               symbol="branch", symbol_map=symbol_map, opacity=0.5).data:
        fig.add_trace(trace, row=1, col=1)

    for trace in px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color="Position",
                               symbol="branch", symbol_map=symbol_map, opacity=0.5).data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(title=f"Model {model_key} — {rep}",
                      scene=dict(xaxis_title=axis_labels["PC1"], yaxis_title=axis_labels["PC2"], zaxis_title=axis_labels["PC3"]),
                      scene2=dict(xaxis_title=axis_labels["PC1"], yaxis_title=axis_labels["PC2"], zaxis_title=axis_labels["PC3"]))
    fig.show()


def plot_pca_all(models_list: list[nn.Module], step: int, lr: float, batch_size: int, seed: int,
                 positions: list[int], rep: str = "r1",
                 ablate: dict[str, bool] | None = None,
                 target: bool = True,
                 show_chars: list[str] | None = ["a", "b", "c", "d", "e"]) -> None:
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 38,
        'axes.labelsize': 34,
        'figure.labelsize': 32,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 22,
        'legend.title_fontsize': 22,
    })
    n = len(models_list)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=True, sharex="col", sharey="col")

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

        vowel_ids = [encode[c] for c in "aeiou"]
        is_vowel = np.isin(all_tokens[:, 0].numpy(), vowel_ids)
        branch = np.repeat(np.where(is_vowel, "reverse", "identity"), n_pos)

        df = pd.DataFrame(projected, columns=["PC1", "PC2"])  # type: ignore
        df["target"] = target_chars
        df["input"] = input_chars
        df["Position"] = pos_array
        df["branch"] = branch

        var = fitted.explained_variance_ratio_
        model_key = model.__class__.__name__[-1].upper()
        print(f"Model {model_key}: PC1={var[0]:.1%}, PC2={var[1]:.1%}")

        ax1, ax2 = axes[0, col], axes[1, col]
        ax1.set_title(f"{model_key}", color=palette[model_key], fontweight="bold", pad=12)

        # top: colored by target or input token
        hue_col = "target" if target else "input"
        plot_df = df if show_chars is None else df.loc[df[hue_col].isin(show_chars)]
        id_df = plot_df[plot_df["branch"] == "identity"]
        rev_df = plot_df[plot_df["branch"] == "reverse"]
        char_order = sorted(plot_df[hue_col].unique())
        pos_order = sorted(plot_df["Position"].unique())
        sns.scatterplot(data=id_df, x="PC1", y="PC2", hue=hue_col, hue_order=char_order,  # type: ignore[arg-type]
                        palette="turbo", alpha=0.5, s=20, marker="o",
                        edgecolor="none", ax=ax1, legend=False, rasterized=True)
        sns.scatterplot(data=rev_df, x="PC1", y="PC2", hue=hue_col, hue_order=char_order,  # type: ignore[arg-type]
                        palette="turbo", alpha=0.5, s=70, marker="X",
                        edgecolor="black", linewidth=0.3, ax=ax1, legend=False, rasterized=True)
        for key, group in plot_df.groupby([hue_col, "branch"]):
            char = key[0]  # type: ignore[index]
            cx, cy = float(group["PC1"].mean()), float(group["PC2"].mean())
            ax1.text(cx, cy, str(char), fontsize=22, fontweight="bold", ha="center", va="center",
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.5,
                               alpha=0.15, boxstyle="circle,pad=0.15"))
        ax1.set_anchor("N")
        ax1.set_xlabel("")
        ax1.tick_params(labelbottom=False)
        ax1.set_ylabel("")

        # bottom: colored by position
        sns.scatterplot(data=id_df, x="PC1", y="PC2", hue="Position", hue_order=pos_order,  # type: ignore[arg-type]
                        palette="turbo", alpha=0.5, s=20, marker="o",
                        edgecolor="none", ax=ax2, legend=False, rasterized=True)
        sns.scatterplot(data=rev_df, x="PC1", y="PC2", hue="Position", hue_order=pos_order,  # type: ignore[arg-type]
                        palette="turbo", alpha=0.5, s=70, marker="X",
                        edgecolor="black", linewidth=0.3, ax=ax2, rasterized=True)
        ax2.set_anchor("N")
        if col == 0:
            positions_sorted = sorted(plot_df["Position"].unique())
            norm = mcolors.Normalize(vmin=min(positions_sorted), vmax=max(positions_sorted))
            cmap = plt.cm.get_cmap("turbo")
            handles = [mpatches.Patch(color=cmap(norm(p)), alpha=0.8, label=str(p)) for p in positions_sorted]
            ax2.legend(handles=handles, title="Position", loc="upper left",
                       handlelength=1.5, handleheight=1.5)
        else:
            ax2.legend_.remove() if ax2.legend_ else None
        ax2.set_xlabel("")
        ax2.set_ylabel("")

    fname = f"pca_all_step{step}_lr{lr}_bs{batch_size}_seed{seed}_{rep}.pdf"
    fig.supxlabel("PC1", x=0.54, fontsize=plt.rcParams['axes.labelsize'])
    fig.supylabel("PC2", y=0.55, fontsize=plt.rcParams['axes.labelsize'])
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.4)
    fig.savefig(_data_dir.parent / "plots" / fname, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_pca_res(models_list: list[nn.Module], step: int, lr: float, batch_size: int, seed: int,
                 positions: list[int],
                 ablate: dict[str, bool] | None = None,
                 show_chars: list[str] | None = ["a", "b", "c", "d", "e"]) -> None:
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 38,
        'axes.labelsize': 34,
        'figure.labelsize': 32,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 22,
        'legend.title_fontsize': 22,
    })
    # row_configs = [("r2", "target"), ("h1", "target"), ("r1", "current")]
    row_configs = [("r2", "target"), ("h1", "target")]
    n = len(models_list)
    fig, axes = plt.subplots(len(row_configs), n, figsize=(4.5 * n, 4 * len(row_configs)),
                             squeeze=False, sharex=False, sharey=False,
                             subplot_kw={"projection": "3d"})

    for col, model in enumerate(models_list):
        result = pca(model, step, lr, batch_size, seed, positions=positions, ablate=ablate)
        model_key = model.__class__.__name__[-1].upper()

        for row, (rep, hue_col) in enumerate(row_configs):
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
            current_tokens = all_tokens[:, positions].numpy().reshape(N * n_pos)
            current_chars = np.array([decode[t] for t in current_tokens])

            vowel_ids = [encode[c] for c in "aeiou"]
            is_vowel = np.isin(all_tokens[:, 0].numpy(), vowel_ids)
            branch = np.repeat(np.where(is_vowel, "reverse", "identity"), n_pos)

            df = pd.DataFrame(projected, columns=["PC1", "PC2", "PC3"])  # type: ignore
            df["target"] = target_chars
            df["input"] = input_chars
            df["current"] = current_chars
            df["Position"] = pos_array
            df["branch"] = branch

            var = fitted.explained_variance_ratio_
            print(f"Model {model_key} [{rep}]: PC1={var[0]:.1%}, PC2={var[1]:.1%}, PC3={var[2]:.1%}")

            ax = axes[row, col]
            if row == 0:
                ax.set_title(f"{model_key}", color=palette[model_key], fontweight="bold", pad=12)

            plot_df = df if show_chars is None else df.loc[df[hue_col].isin(show_chars)]
            id_df = plot_df[plot_df["branch"] == "identity"]
            rev_df = plot_df[plot_df["branch"] == "reverse"]
            char_order = sorted(plot_df[hue_col].unique())
            turbo_colors = sns.color_palette("turbo", len(char_order))
            color_map = {c: turbo_colors[i] for i, c in enumerate(char_order)}
            rev_colors = [color_map[c] for c in rev_df[hue_col]]
            id_colors = [color_map[c] for c in id_df[hue_col]]
            id_sc = ax.scatter(id_df["PC1"], id_df["PC2"], id_df["PC3"],
                               c=id_colors, alpha=0.5, s=80, marker="o",
                               edgecolor="black", linewidth=0.15)
            rev_sc = ax.scatter(rev_df["PC1"], rev_df["PC2"], rev_df["PC3"],
                                c=rev_colors, alpha=0.5, s=80, marker="o",
                                edgecolor="black", linewidth=0.15)
            id_sc.set_rasterized(True)
            rev_sc.set_rasterized(True)
            for char in plot_df[hue_col].unique():
                char_df = plot_df[plot_df[hue_col] == char]
                id_g = char_df[char_df["branch"] == "identity"]
                ref_g = id_g if len(id_g) > 0 else char_df[char_df["branch"] == "reverse"]
                if len(ref_g) == 0:
                    continue
                cx = float(ref_g["PC1"].mean())
                cy = float(ref_g["PC2"].mean())
                cz = float(ref_g["PC3"].mean())
                ax.text(cx, cy, cz, str(char), fontsize=32, fontweight="bold",
                        ha="center", va="center")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")  # type: ignore[attr-defined]
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.zaxis.set_major_locator(MaxNLocator(3))  # type: ignore[attr-defined]
            ax.xaxis.pane.set_facecolor("white")
            ax.yaxis.pane.set_facecolor("white")
            ax.zaxis.pane.set_facecolor("white")  # type: ignore[attr-defined]
            ax.xaxis.pane.set_edgecolor("lightgray")
            ax.yaxis.pane.set_edgecolor("lightgray")
            ax.zaxis.pane.set_edgecolor("lightgray")  # type: ignore[attr-defined]
            # ax.view_init(elev=15, azim=-70)                                                  

            if rep == 'h1':
                rep = 'A'
            if col == 0:
                ax.text2D(-0.12, 0.5, f"{rep.upper()}", transform=ax.transAxes,
                          fontsize=plt.rcParams['axes.labelsize'],
                          ha="center", va="center", rotation=90)

    fname = f"pca_res_step{step}_lr{lr}_bs{batch_size}_seed{seed}.pdf"
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.10, wspace=-0.35)
    fig.savefig(_data_dir.parent / "plots" / fname, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()


# --- Ablations ---

def plot_ablations(step: int, lr: float, batch_size: int, test: bool = False) -> None:
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 34,
        'xtick.labelsize': 30,
        'ytick.labelsize': 28,
        'legend.fontsize': 28,
        'legend.title_fontsize': 22,
    })
    model_configs: list[tuple[str, nn.Module, list[tuple[str, dict[str, bool]]]]] = [
        ("F", ModelF(), [
            ("_",   {'h1': False, 'prog_out': False}),
            ("A",   {'h1': True,  'prog_out': False}),
            ("R",   {'h1': False, 'prog_out': True}),
            ("A+R", {'h1': True,  'prog_out': True}),
        ]),
        ("U", ModelU(), [
            ("_",   {'h1': False, 'prog_out': False}),
            ("A",   {'h1': True,  'prog_out': False}),
            ("R",   {'h1': False, 'prog_out': True}),
            ("A+R", {'h1': True,  'prog_out': True}),
        ]),
        ("T", ModelT(), [
            ("_",   {'h1': False, 'prog_out': False}),
            ("A",   {'h1': True,  'prog_out': False}),
            ("R",   {'h1': False, 'prog_out': True}),
            ("A+R", {'h1': True,  'prog_out': True}),
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
        model.to(DEVICE)
        for seed in _seeds:
            ckpt_path = _data_dir / "checkpoints" / model_name / f"checkpoint_step{step}_seed{seed}_lr{lr}_bs{batch_size}.pt"
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["state_dict"])

            for label, ablate in conditions:
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        logits = model(x, ablate=ablate)
                        preds = logits[:, 15:, :].argmax(dim=-1)
                        targets = y[:, 15:]
                        correct += (preds == targets).sum().item()
                        total += targets.numel()
                rows.append({"condition": label, "model": model_label, "accuracy": correct / total})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    hatches = {"F": "", "U": "/", "T": "--", "I": "||"}
    sns.barplot(data=df, x="condition", y="accuracy", hue="model",
                order=all_conditions, hue_order=hue_order,
                palette=palette, errorbar="sd", capsize=0.4, ax=ax, width=0.65,
                err_kws={"color": "black", "linewidth": 2.5, "solid_capstyle": "projecting"})
    for container, model_label in zip(ax.containers, hue_order):
        for bar in container:
            bar.set_hatch(hatches[model_label])
    ax.axhline(1 / 26, color="black", linestyle="--", linewidth=3.5, label="Ch")
    ax.set_xlabel("Ablation", labelpad=20)
    ax.set_ylabel("Accuracy", labelpad=20)
    ax.set_ylim(0, 1)
    legend = ax.legend(title="")
    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        label = text.get_text()
        if label in palette:
            text.set_color(palette[label])
            if isinstance(handle, mpatches.Patch):
                handle.set_hatch(hatches[label])
        elif label == "Ch":
            text.set_fontsize(24)
    fig.tight_layout()
    fname = f"ablations_step{step}_lr{lr}_bs{batch_size}.pdf"
    fig.savefig(_data_dir.parent / "plots" / fname, bbox_inches='tight', pad_inches=0)
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