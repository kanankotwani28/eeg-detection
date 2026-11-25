"""Build every thesis figure and the value table (values.json) from artifacts/results.
All numbers in the thesis come from here (CLAUDE.md rule 10). Run: python thesis/build_figures.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "artifacts" / "results"
FIG = ROOT / "thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TEAL, TEAL_L, PEACH, PEACH_E, INK, GREY = "#2f5d62", "#e3ebe8", "#f8ebe3", "#d9a184", "#1f2d3a", "#8a8f94"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- diagram helpers
def box(ax, x, y, w, h, title, sub="", fill="white", edge=TEAL, bold=True, fs=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fill, ec=edge, lw=0.9))
    if sub:
        ax.text(x, y + h * 0.16, title, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None,
                color=INK)
        ax.text(x, y - h * 0.2, sub, ha="center", va="center", fontsize=fs - 1.6, color=INK, linespacing=1.15)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, color=INK,
                linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, label="", color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.8, color=color))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, label, fontsize=6.5, ha="center", va="bottom", color=INK)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- explanatory figures
def fig_rhythms():
    t = np.linspace(0, 2, 1000)
    bands = [("Delta (1-4 Hz)", 2), ("Theta (4-8 Hz)", 6), ("Alpha (8-13 Hz)", 10), ("Beta (13-30 Hz)", 20)]
    rng = np.random.default_rng(3)
    mixed = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * a for (_, f), a in zip(bands, [1.0, 0.6, 0.9, 0.3])) \
        + 0.25 * rng.standard_normal(len(t))
    fig, axes = plt.subplots(5, 1, figsize=(6.2, 4.2), sharex=True)
    rows = [("Mixed scalp signal", mixed)] + [(n, np.sin(2 * np.pi * f * t)) for n, f in bands]
    cols = [INK, TEAL, "#4f8a8b", "#7fb0a8", "#c47f5a"]
    for ax, (n, y), c in zip(axes, rows, cols, strict=True):
        ax.plot(t, y, color=c, lw=0.8)
        ax.set_yticks([])
        ax.set_ylabel(n, rotation=0, ha="right", va="center", fontsize=7.5)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    save(fig, "fig_rhythms")


def fig_gap():
    fig, ax = canvas(6.4, 2.2)
    box(ax, 0.16, 0.68, 0.27, 0.3, "Single-site studies", "segment splits, small samples,\nreported accuracy > 95%")
    box(ax, 0.5, 0.68, 0.27, 0.3, "Open question", "how much accuracy do federation\nand participant-level DP cost?",
        fill=PEACH, edge=PEACH_E)
    box(ax, 0.84, 0.68, 0.27, 0.3, "This project", "leakage-safe benchmark:\nFL + DP + LiRA + external test",
        fill=TEAL_L)
    arrow(ax, 0.295, 0.68, 0.365, 0.68)
    arrow(ax, 0.635, 0.68, 0.705, 0.68)
    box(ax, 0.5, 0.18, 0.86, 0.22, "Needed together:  subject-level splits  ·  privacy accounting per participant  ·  "
        "attack-based audit  ·  independent external data  ·  honest reporting", bold=False, fs=7.8)
    arrow(ax, 0.5, 0.29, 0.5, 0.53)
    save(fig, "fig_gap")


def fig_architecture():
    fig, ax = canvas(6.4, 3.6)
    box(ax, 0.14, 0.86, 0.22, 0.14, "TDBRAIN (BIDS)", "BDF, 26 EEG ch, 500 Hz")
    box(ax, 0.43, 0.86, 0.24, 0.14, "Data layer", "participant index,\nfrozen splits", fill=TEAL_L)
    box(ax, 0.74, 0.86, 0.26, 0.14, "Preprocessing layer", "19 ch, 1-40 Hz, CAR,\n256 Hz, 4 s epochs", fill=TEAL_L)
    box(ax, 0.74, 0.56, 0.26, 0.16, "Model layer", "A: band power + FAA -> LR\nB: EEGNet-8,2 (GroupNorm)", fill=TEAL_L)
    box(ax, 0.43, 0.56, 0.24, 0.16, "Regime layer", "central · local · FedAvg\n· FedAvg + participant DP", fill=TEAL_L)
    box(ax, 0.14, 0.56, 0.22, 0.16, "Evaluation layer", "metrics + bootstrap CI,\nLiRA audit, Mumtaz", fill=TEAL_L)
    for a, b in [((0.25, 0.86), (0.31, 0.86)), ((0.55, 0.86), (0.61, 0.86)), ((0.74, 0.79), (0.74, 0.64)),
                 ((0.61, 0.56), (0.55, 0.56)), ((0.31, 0.56), (0.25, 0.56))]:
        arrow(ax, *a, *b)
    box(ax, 0.5, 0.27, 0.9, 0.13, "Tracking and reporting", "MLflow (one run per experiment, fold and seed)  ->  "
        "export  ->  REPORT.md and Streamlit dashboard", fs=8)
    box(ax, 0.5, 0.08, 0.9, 0.12, "Data governance", "raw EEG stays in the local data folder, is never committed; "
        "only aggregate results are reported", fill=PEACH, edge=PEACH_E, fs=8)
    arrow(ax, 0.14, 0.48, 0.14, 0.33)
    arrow(ax, 0.5, 0.21, 0.5, 0.135)
    save(fig, "fig_architecture")


def fig_pipeline():
    fig, ax = canvas(4.6, 5.2)
    steps = ["1. Load first-session restEC recording (BDF)", "2. Drop non-EEG channels, rename T3/T4/T5/T6",
             "3. Pick 19 shared channels in fixed order", "4. Volts -> microvolts; FIR 1-40 Hz; 50 Hz notch",
             "5. Within-recording bad-channel interpolation (counted)", "6. Common average reference; resample to 256 Hz",
             "7. 4 s epochs; reject > 150 uV peak-to-peak or flat", "8. Keep first 30 clean epochs",
             "9. Cache epochs (N, 19, 1024) and QC record"]
    ys = np.linspace(0.95, 0.05, len(steps) + 1)
    ys = np.delete(ys, 8)
    for i, (s, y) in enumerate(zip(steps, ys, strict=True)):
        box(ax, 0.42, y, 0.78, 0.058, s, fill=TEAL_L if 0 < i < 8 else "white", bold=False, fs=7.6)
        if i < len(steps) - 1 and i != 7:
            arrow(ax, 0.42, y - 0.029, 0.42, ys[i + 1] + 0.029)
    # decision between 8 and 9
    yd = (ys[7] + ys[8]) / 2
    ax.plot([0.42, 0.42], [ys[7] - 0.029, yd + 0.03], color=INK, lw=0.8)
    arrow(ax, 0.42, yd - 0.03, 0.42, ys[8] + 0.029)
    ax.text(0.44, yd - 0.045, "yes", fontsize=6.5)
    ax.add_patch(Polygon([[0.42, yd + 0.03], [0.53, yd], [0.42, yd - 0.03], [0.31, yd]], closed=True, fc="white",
                         ec=INK, lw=0.8))
    ax.text(0.42, yd, ">= 10 clean?", ha="center", va="center", fontsize=6.5)
    box(ax, 0.84, yd, 0.26, 0.06, "Excluded,\nreason logged", fill=PEACH, edge=PEACH_E, bold=False, fs=6.8)
    arrow(ax, 0.53, yd, 0.71, yd, "no")
    save(fig, "fig_pipeline")


def fig_regimes():
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.3))
    titles = ["Centralised", "Local only", "Federated (FedAvg)", "Federated + DP"]
    for ax, t in zip(axes, titles, strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(t, fontsize=8.5, fontweight="bold", color=INK)
    a = axes[0]
    for i in range(4):
        box(a, 0.15 + 0.23 * i, 0.2, 0.18, 0.14, f"C{i + 1}", bold=False, fs=7)
        arrow(a, 0.15 + 0.23 * i, 0.27, 0.5, 0.62)
    box(a, 0.5, 0.72, 0.6, 0.18, "pooled data\none model", fill=PEACH, edge=PEACH_E, bold=False, fs=7)
    a = axes[1]
    for i in range(4):
        box(a, 0.15 + 0.23 * i, 0.2, 0.18, 0.14, f"C{i + 1}", bold=False, fs=7)
        box(a, 0.15 + 0.23 * i, 0.62, 0.18, 0.14, f"M{i + 1}", fill=TEAL_L, bold=False, fs=7)
        arrow(a, 0.15 + 0.23 * i, 0.27, 0.15 + 0.23 * i, 0.55)
    a.text(0.5, 0.86, "no sharing", ha="center", fontsize=7)
    for k, a in enumerate(axes[2:]):
        box(a, 0.5, 0.78, 0.5, 0.16, "server\nFedAvg", fill=TEAL_L, bold=False, fs=7)
        for i in range(4):
            x = 0.15 + 0.23 * i
            box(a, x, 0.2, 0.18, 0.14, f"C{i + 1}", bold=False, fs=7)
            arrow(a, x, 0.27, 0.5, 0.7, color=TEAL)
        if k == 1:
            a.text(0.5, 0.45, "clip per participant\n+ Gaussian noise", ha="center", fontsize=6.3, color="#9b4d2b")
    fig.text(0.5, -0.02, "Raw EEG of a client never leaves that client in the local, federated and DP regimes",
             ha="center", fontsize=7.5)
    save(fig, "fig_regimes")


def fig_montage():
    import mne

    from src.utils.paths import standard_montage

    chs = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
           "O1", "O2"]
    extra = ["FC3", "FCz", "FC4", "CP3", "CPz", "CP4", "Oz"]
    info = mne.create_info(chs + extra, 256.0, "eeg")
    info.set_montage(standard_montage())
    pos = {c: info["chs"][i]["loc"][:2] for i, c in enumerate(chs + extra)}
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    r = 0.105
    ax.add_patch(plt.Circle((0, 0.005), r, fill=False, color=INK, lw=0.9))
    ax.add_patch(Polygon([[-0.012, r + 0.003], [0, r + 0.016], [0.012, r + 0.003]], fill=False, ec=INK, lw=0.9))
    for c, (x, y) in pos.items():
        shared = c in chs
        ax.add_patch(plt.Circle((x, y), 0.0085, fc=TEAL_L if shared else "white", ec=TEAL if shared else GREY, lw=0.8))
        ax.text(x, y, c, ha="center", va="center", fontsize=5.2, color=INK if shared else GREY)
    ax.set_xlim(-0.125, 0.125)
    ax.set_ylim(-0.12, 0.13)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "fig_montage")


def fig_eegnet():
    fig, ax = canvas(6.6, 2.0)
    blocks = [("Input", "1 x 19 x 1024"), ("Temporal conv", "8 filters, 1 x 128\n+ GroupNorm"),
              ("Depthwise spatial", "D = 2 -> 16 maps\n19 x 1, max-norm 1"), ("ELU, AvgPool 4", "Dropout 0.5"),
              ("Separable conv", "16 maps, 1 x 16\n+ GroupNorm"), ("ELU, AvgPool 8", "Dropout 0.5"),
              ("Dense", "2 logits\nmax-norm 0.25")]
    xs = np.linspace(0.07, 0.93, len(blocks))
    for i, ((t, s), x) in enumerate(zip(blocks, xs, strict=True)):
        box(ax, x, 0.55, 0.125, 0.62, t, s, fill=TEAL_L if 0 < i < 6 else "white", fs=7.2)
        if i < len(blocks) - 1:
            arrow(ax, x + 0.0625, 0.55, xs[i + 1] - 0.0625, 0.55)
    ax.text(0.5, 0.06, "Per-recording, per-channel z-scored 4 s epochs; person-level probability = mean of epoch "
            "probabilities", ha="center", fontsize=7.2)
    save(fig, "fig_eegnet")


def fig_fl_dp():
    fig, ax = canvas(6.4, 3.1)
    box(ax, 0.5, 0.88, 0.42, 0.14, "Server (honest-but-curious)", "sends global weights w_t;\nweighted FedAvg; "
        "keeps best round", fill=TEAL_L)
    for i, x in enumerate([0.14, 0.38, 0.62, 0.86]):
        box(ax, x, 0.36, 0.21, 0.34, f"Client {i + 1}", "Poisson-sample\nparticipants (q = 0.25)\none epoch each\n"
            "clip each grad to C = 1\nsum + N(0, s^2 C^2)\n/ expected batch", fill="white", fs=7.2)
        arrow(ax, 0.5 - 0.15 + 0.1 * i, 0.81, x, 0.535, color=TEAL)
        arrow(ax, x + 0.02, 0.535, 0.5 - 0.13 + 0.1 * i, 0.81, color="#9b4d2b")
    ax.text(0.98, 0.66, "noisy updates", ha="right", fontsize=7, color="#9b4d2b")
    ax.text(0.02, 0.66, "global weights", ha="left", fontsize=7, color=TEAL)
    box(ax, 0.5, 0.05, 0.94, 0.08, "RDP accountant (Opacus): eps from (sigma, q, steps of each client); reported "
        "eps = max over clients; privacy unit = participant", fill=PEACH, edge=PEACH_E, bold=False, fs=7.3)
    save(fig, "fig_fl_dp")


def fig_lira():
    fig, ax = canvas(6.4, 2.4)
    box(ax, 0.12, 0.62, 0.2, 0.32, "Pool", "467 labelled\nT1 participants")
    box(ax, 0.38, 0.62, 0.24, 0.32, "S models", "each on a random 50%\n(same recipe as target)",
        fill=TEAL_L)
    box(ax, 0.64, 0.62, 0.22, 0.32, "Per participant", "Gaussians of IN and\nOUT confidence", fill=TEAL_L)
    box(ax, 0.88, 0.62, 0.2, 0.32, "Likelihood ratio", "on the held-out\ntarget's score", fill=PEACH, edge=PEACH_E)
    for a, b in [(0.22, 0.26), (0.5, 0.53), (0.75, 0.78)]:
        arrow(ax, a, 0.62, b, 0.62)
    ax.text(0.5, 0.12, "Each model plays the target once (leave-one-out). Report ROC, AUC and TPR at 1% / 5% FPR. "
            "Score = mean over epochs of logit(p_correct).", ha="center", fontsize=7.3)
    save(fig, "fig_lira")


def fig_splits(c):
    fig, ax = canvas(6.4, 2.4)
    for k in range(5):
        x0 = 0.05 + 0.18 * k
        ax.add_patch(FancyBboxPatch((x0, 0.62), 0.16, 0.2, boxstyle="round,pad=0.005", fc=PEACH if k == 0 else TEAL_L,
                                    ec=TEAL, lw=0.8))
        ax.text(x0 + 0.08, 0.72, f"fold {k + 1}" + ("\n(test)" if k == 0 else ""), ha="center", va="center", fontsize=7)
    ax.text(0.5, 0.92, f"Outer: StratifiedGroupKFold(5), by participant, stratified by label x age band "
            f"(T1 n = {c['T1_n']})", ha="center", fontsize=7.6)
    box(ax, 0.3, 0.27, 0.3, 0.18, "Inner 80 / 20 split", "train / validation\n(C, threshold, early stopping)",
        fill="white", fs=7.4)
    box(ax, 0.72, 0.27, 0.42, 0.18, "Training participants -> K = 4 clients", "IID (random) or label-skew\n"
        "(Dirichlet a = 0.5, >= 5 per class)", fill="white", fs=7.4)
    arrow(ax, 0.6, 0.6, 0.35, 0.37)
    arrow(ax, 0.6, 0.6, 0.72, 0.37)
    ax.text(0.5, 0.02, "Saved once to splits.json with a SHA-256 hash; no participant is in two sets or two clients",
            ha="center", fontsize=7.2)
    save(fig, "fig_splits")


def fig_cohort(c):
    fig, ax = canvas(5.4, 4.0)
    rows = [("Participant table (TDBRAIN V3)", f"{c['rows']} session rows"),
            ("First session per participant", f"n = {c['first_session']}"),
            ("With an eyes-closed recording", f"n = {c['with_restEC']}"),
            ("Labelled (UNKNOWN / missing excluded)", f"n = {c['labelled']}  (excluded {c['unknown']})"),
            ("Adults 18-65 with a T1 or T2 label", f"T1 n = {c['T1_n']}  ·  T2 n = {c['T2_n']}")]
    ys = np.linspace(0.9, 0.12, len(rows))
    for i, ((t, s), y) in enumerate(zip(rows, ys, strict=True)):
        box(ax, 0.5, y, 0.7, 0.13, t, s, fill=TEAL_L if i == len(rows) - 1 else "white", fs=8)
        if i < len(rows) - 1:
            arrow(ax, 0.5, y - 0.065, 0.5, ys[i + 1] + 0.065)
    save(fig, "fig_cohort")


def fig_roadmap():
    fig, ax = canvas(6.2, 1.9)
    box(ax, 0.16, 0.66, 0.27, 0.4, "Now", "benchmark on TDBRAIN;\nLR complete, EEGNet\npreliminary", fill=TEAL_L)
    box(ax, 0.5, 0.66, 0.27, 0.4, "Next", "full EEGNet runs (3 seeds,\n100 epochs), E07-E08,\nE12, E14-E20")
    box(ax, 0.84, 0.66, 0.27, 0.4, "Then", "real multi-site data\n(e.g. AIIMS), secure\naggregation")
    arrow(ax, 0.295, 0.66, 0.365, 0.66)
    arrow(ax, 0.635, 0.66, 0.705, 0.66)
    box(ax, 0.5, 0.12, 0.7, 0.18, "Only after clinical validation: any use beyond research", fill=PEACH, edge=PEACH_E,
        bold=False, fs=7.8)
    save(fig, "fig_roadmap")


# ---------------------------------------------------------------- result figures
def lab(r):
    names = {"E01": "E01 shuffled", "E02": "E02 band power + FAA", "E16": "E16 FAA only", "E22": "E22 age only",
             "E03": "E03 T2 (MDD vs other)", "E04": "E04 EEGNet (fold 0)", "E05": "E05 local-only",
             "E06": "E06 FedAvg", "E09": "E09 FedAvg + DP"}
    v = "" if r["variant"] in ("default", "T1") else f" {r['variant'].replace('_', '-')}"
    e = "" if r["eps"] == "inf" else f" eps={r['eps']}"
    return f"{names.get(r['experiment'], r['experiment'])}{v}{e} [{r['model']}] (n={int(r['n_participants'])})"


def fig_main(s):
    keep = s[s["experiment"].isin(["E01", "E02", "E16", "E22", "E03", "E04", "E05", "E06"])].copy()
    keep = keep[~((keep["experiment"] == "E01") & (keep["model"] == "eegnet"))]
    keep["label"] = keep.apply(lab, axis=1)
    keep = keep.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(keep) + 0.8))
    y = np.arange(len(keep))
    for m, col in (("balanced_accuracy", TEAL), ("auc", "#c47f5a")):
        off = -0.15 if m == "balanced_accuracy" else 0.15
        ax.errorbar(keep[m], y + off, xerr=[keep[m] - keep[m + "_lo"], keep[m + "_hi"] - keep[m]], fmt="o", ms=3.5,
                    color=col, capsize=2, lw=0.9, label="Balanced accuracy" if off < 0 else "AUC")
    ax.axvline(0.5, ls="--", c=GREY, lw=0.8)
    ax.set_yticks(y, keep["label"], fontsize=7)
    ax.set_xlabel("Participant-level score (95% bootstrap CI)")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    save(fig, "fig_main_results")


def fig_roc(roc):
    sel = [("E02", "default", "logreg", "inf", "E02 LR central"), ("E06", "iid", "logreg", "inf", "E06 LR FedAvg"),
           ("E09", "default", "logreg", "8", "E09 LR FedAvg + DP eps=8"), ("E22", "default", "logreg", "inf",
                                                                           "E22 age only"),
           ("E16", "default", "logreg", "inf", "E16 FAA only")]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    cols = [TEAL, "#4f8a8b", "#c47f5a", GREY, "#b9b0a0"]
    for (e, v, m, eps, name), c in zip(sel, cols, strict=True):
        g = roc[(roc.experiment == e) & (roc.variant == v) & (roc.model == m) & (roc.eps == eps)]
        if len(g):
            ax.plot(g.fpr, g.tpr, color=c, lw=1.1, label=name)
    ax.plot([0, 1], [0, 1], ls="--", c=GREY, lw=0.7)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    save(fig, "fig_roc")


def fig_fl(s):
    order = [("E02", "default", "Centralised"), ("E06", "iid", "FedAvg IID"), ("E06", "label_skew", "FedAvg label-skew"),
             ("E06", "iid_dropout", "FedAvg IID + dropout"), ("E05", "iid", "Local only (mean client)"),
             ("E05", "label_skew", "Local only, skewed (mean)")]
    rows = [s[(s.experiment == e) & (s.variant == v) & (s.model == "logreg")].iloc[0].to_dict() | {"name": n}
            for e, v, n in order]
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = np.arange(len(d))
    ax.bar(x, d.balanced_accuracy, color=[TEAL] + ["#4f8a8b"] * 3 + ["#c47f5a"] * 2, width=0.6,
           yerr=[d.balanced_accuracy - d.balanced_accuracy_lo, d.balanced_accuracy_hi - d.balanced_accuracy],
           capsize=3, error_kw={"lw": 0.8})
    if "worst_balanced_accuracy" in d:
        w = d.worst_balanced_accuracy
        ax.scatter(x[w.notna()], w[w.notna()], marker="v", color=INK, zorder=3, s=18, label="worst client")
        ax.legend(fontsize=7, frameon=False)
    ax.axhline(0.5, ls="--", c=GREY, lw=0.8)
    ax.set_xticks(x, d.name, fontsize=7, rotation=12)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("Balanced accuracy (95% CI)")
    ax.set_title(f"Logistic regression, T1, n = {int(d.n_participants.max())} participants", fontsize=8)
    save(fig, "fig_fl")


def fig_privacy(s, lira):
    d = s[(s.experiment == "E09") & (s.model == "logreg")].copy()
    d["k"] = d.eps.map(lambda e: 99 if e == "inf" else float(e))
    d = d.sort_values("k")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axes[0]
    xl = [("inf" if e == "inf" else e) for e in d.eps]
    x = np.arange(len(d))
    ax.errorbar(x, d.balanced_accuracy, yerr=[d.balanced_accuracy - d.balanced_accuracy_lo,
                                               d.balanced_accuracy_hi - d.balanced_accuracy], fmt="o-", color=TEAL,
                capsize=3, label="Balanced accuracy")
    ax.errorbar(x + 0.06, d.auc, yerr=[d.auc - d.auc_lo, d.auc_hi - d.auc], fmt="s--", color="#c47f5a", capsize=3,
                label="AUC", ms=3.5)
    ax.set_xticks(x, ["∞ (no DP)" if v == "inf" else v for v in xl])
    ax.set_xlabel("Participant-level ε (δ = 1e-5)")
    ax.axhline(0.5, ls=":", c=GREY, lw=0.8)
    ax.set_title(f"Utility (n = {int(d.n_participants.max())})", fontsize=8)
    ax.legend(fontsize=6.8, frameon=False)
    ax = axes[1]
    l2 = lira[(lira.model == "logreg") & (lira.regime == "federated")].copy()
    l2["k"] = l2.eps.map(lambda e: 99 if e == "inf" else float(e))
    l2 = l2.sort_values("k")
    ax.plot(np.arange(len(l2)), l2.auc, "o-", color="#9b4d2b")
    c = lira[(lira.model == "logreg") & (lira.regime == "central")]
    if len(c):
        ax.axhline(float(c.auc.iloc[0]), ls="--", color=INK, lw=0.8, label="centralised, no DP")
        ax.legend(fontsize=6.8, frameon=False)
    ax.axhline(0.5, ls=":", c=GREY, lw=0.8)
    ax.set_xticks(np.arange(len(l2)), ["∞ (no DP)" if v == "inf" else v for v in l2.eps])
    ax.set_xlabel("Participant-level ε")
    ax.set_ylabel("LiRA attack AUC")
    ax.set_title(f"Membership leakage ({int(l2.n_nonmember_evals.iloc[0])} non-member evaluations)", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_privacy")


def fig_lira_roc(lroc):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    cols = {"central/inf": INK, "federated/inf": TEAL, "federated/8": "#4f8a8b", "federated/3": "#c47f5a",
            "federated/1": "#d9a184"}
    for (r, e), g in lroc[lroc.model == "logreg"].groupby(["regime", "eps"]):
        g = g[(g.fpr > 0) & (g.tpr > 0)]
        k = f"{r}/{e}"
        ax.plot(g.fpr, g.tpr, lw=1, color=cols.get(k, GREY), label=f"{r}, ε = {'∞' if e == 'inf' else e}")
    ax.plot([1e-4, 1], [1e-4, 1], ls="--", c=GREY, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    save(fig, "fig_lira_roc")


def fig_external(s, ext):
    ref = {"LR-central": ("E02", "default", "inf"), "LR-FedAvg": ("E06", "iid", "inf"),
           "LR-FedAvg-DP-eps8": ("E09", "default", "8")}
    e = ext[ext.kind == "external"]
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    for i, (_, r) in enumerate(e.iterrows()):
        ex, v, eps = ref[r.source]
        q = s[(s.experiment == ex) & (s.variant == v) & (s.model == "logreg") & (s.eps == eps)].iloc[0]
        ax.errorbar(q.auc, i + 0.12, xerr=[[q.auc - q.auc_lo], [q.auc_hi - q.auc]], fmt="o", color=TEAL, capsize=3,
                    label="TDBRAIN CV (AUC)" if i == 0 else None)
        ax.errorbar(r.auc, i - 0.12, xerr=[[r.auc - r.auc_lo], [r.auc_hi - r.auc]], fmt="D", color="#c47f5a",
                    capsize=3, label=f"Mumtaz external (AUC, n = {int(r.n)})" if i == 0 else None)
    ax.axvline(0.5, ls="--", c=GREY, lw=0.8)
    ax.set_yticks(range(len(e)), e.source)
    ax.set_xlabel("AUC (95% bootstrap CI)")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    save(fig, "fig_external")


def fig_psd():
    p = pd.read_parquet(RES / "psd_tdbrain_restEC.parquet")
    p = p[p.group.isin(["MDD", "HEALTHY", "ADHD", "OCD", "SMC"]) & (p.freq <= 40) & (p.freq >= 1)]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    cols = {"MDD": "#c47f5a", "HEALTHY": TEAL, "ADHD": "#4f8a8b", "OCD": GREY, "SMC": "#b9b0a0"}
    for g, d in p.groupby("group"):
        m = d.groupby("freq").log10_power.agg(["mean", "sem"])
        n = d.participant_id.nunique()
        ax.plot(m.index, m["mean"], color=cols[g], lw=1.1, label=f"{g} (n = {n})")
        ax.fill_between(m.index, m["mean"] - 1.96 * m["sem"], m["mean"] + 1.96 * m["sem"], color=cols[g], alpha=0.15)
    for f in (4, 8, 13, 30):
        ax.axvline(f, ls=":", c=GREY, lw=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log10 power (µV²/Hz)")
    ax.legend(fontsize=6.8, frameon=False, ncol=2)
    save(fig, "fig_psd")


def fig_explain():
    g = pd.read_parquet(RES / "explain_lr_grouped.parquet")
    if g.empty:
        return
    piv = g.pivot_table(index="band", columns="region", values="coef_mean")
    order_b = [b for b in ["delta", "theta", "alpha", "beta", "FAA"] if b in piv.index]
    order_r = [r for r in ["frontal", "central", "temporal", "parietal", "occipital"] if r in piv.columns]
    piv = piv.loc[order_b, order_r]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    v = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(order_r)), order_r)
    ax.set_yticks(range(len(order_b)), order_b)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if not np.isnan(piv.values[i, j]):
                ax.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean standardised coefficient")
    ax.spines[:].set_visible(False)
    save(fig, "fig_explain")


def main():
    s = pd.read_parquet(RES / "summary_table.parquet")
    lira = pd.read_parquet(RES / "lira.parquet")
    lroc = pd.read_parquet(RES / "lira_roc.parquet")
    ext = pd.read_parquet(RES / "external.parquet")
    roc = pd.read_parquet(RES / "roc.parquet")
    comps = pd.read_parquet(RES / "comparisons.parquet")
    cohort = json.loads((RES / "cohort.json").read_text())
    qc = json.loads((RES / "qc.json").read_text())
    meta = json.loads((RES / "summary.json").read_text())
    fc = cohort["flowchart"]
    c = {"rows": cohort["n_rows_participants_tsv"], "first_session": fc["first_session"], "with_restEC": fc["with_restEC"],
         "labelled": fc["labelled_not_blinded"], "unknown": cohort["dropped_unknown"], "T1_n": fc["T1_final"],
         "T2_n": fc["T2_final"]}
    for f in (fig_rhythms, fig_gap, fig_architecture, fig_pipeline, fig_regimes, fig_montage, fig_eegnet, fig_fl_dp,
              fig_lira, fig_roadmap, fig_psd, fig_explain):
        f()
    fig_splits(c)
    fig_cohort(c)
    fig_main(s)
    fig_roc(roc)
    fig_fl(s)
    fig_privacy(s, lira)
    fig_lira_roc(lroc)
    fig_external(s, ext)
    # ------------------------------------------------------------ values for the text and tables
    # aggregate values only: per-participant QC entries (keyed by participant ID) are dropped
    qc_agg = {k: v for k, v in qc.items() if not isinstance(v, dict | list) or k == "psd_check"}
    V = {"cohort": cohort, "qc": qc_agg, "c": c, "git_commit": meta.get("git_commit"), "split_sha": meta.get("split_sha256"),
         "n_runs": meta.get("n_runs"), "packages": meta.get("packages")}
    V["summary"] = json.loads(s.to_json(orient="records"))
    V["lira"] = json.loads(lira.to_json(orient="records"))
    V["external"] = json.loads(ext.to_json(orient="records"))
    V["comparisons"] = json.loads(comps.to_json(orient="records"))
    (ROOT / "thesis" / "values.json").write_text(json.dumps(V, indent=1, default=str))
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
