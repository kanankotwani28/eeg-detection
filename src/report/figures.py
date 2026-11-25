"""Static figures for REPORT.md (matplotlib). The dashboard draws its own Plotly versions."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.paths import ROOT  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"


def privacy_utility(summary: pd.DataFrame, path=None):
    d = summary[summary["experiment"] == "E09"].copy()
    if d.empty:
        return None
    d["eps_num"] = d["eps"].map(lambda e: np.inf if e == "inf" else float(e))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    finite = [e for e in sorted(d["eps_num"].unique()) if np.isfinite(e)]
    xpos = {e: i for i, e in enumerate(finite + [np.inf])}
    for m, g in d.groupby("model"):
        g = g.sort_values("eps_num")
        x = [xpos[e] for e in g["eps_num"]]
        y = g["balanced_accuracy"].to_numpy()
        err = np.vstack([y - g["balanced_accuracy_lo"], g["balanced_accuracy_hi"] - y])
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=f"{m} (n={int(g['n_participants'].max())})")
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xticks(list(xpos.values()), [f"{e:g}" if np.isfinite(e) else "∞ (no DP)" for e in xpos])
    ax.set_xlabel("participant-level ε (δ = 1e-5)")
    ax.set_ylabel("balanced accuracy (95% CI)")
    ax.set_title("Privacy-utility trade-off (FedAvg + participant-level DP)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = path or FIG_DIR / "privacy_utility.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main_results(summary: pd.DataFrame, path=None):
    d = summary[(summary["experiment"].isin(["E01", "E02", "E03", "E04", "E05", "E06", "E12", "E16", "E22"]))].copy()
    if d.empty:
        return None
    d["label"] = d["experiment"] + " " + d["variant"].replace("default", "") + " " + d["model"]
    d = d.sort_values("label")
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(d) + 1))
    y = np.arange(len(d))
    ba = d["balanced_accuracy"].to_numpy()
    ax.errorbar(ba, y, xerr=np.vstack([ba - d["balanced_accuracy_lo"], d["balanced_accuracy_hi"] - ba]), fmt="o", capsize=2)
    ax.axvline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_yticks(y, d["label"], fontsize=7)
    ax.set_xlabel("participant-level balanced accuracy (95% bootstrap CI)")
    fig.tight_layout()
    path = path or FIG_DIR / "main_results.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
