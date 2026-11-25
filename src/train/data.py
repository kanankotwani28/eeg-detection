"""Glue between the frozen splits, the epoch cache and the feature table for one experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.splits import get_fold, load_splits
from src.features.spectral import feature_table, participant_relative
from src.preprocess.pipeline import cache_paths, load_cache, participant_epochs, run_dataset, tdbrain_items
from src.utils.config import deep_merge, load_config
from src.utils.logging import get_logger
from src.utils.paths import PARTICIPANTS_FILE, SPLITS_FILE

log = get_logger(__name__)

DATASET_VARIANTS = {
    "tdbrain": {},
    "tdbrain-ica": {"ica": True},
    "tdbrain-noref": {"reference": "none"},
    "tdbrain26": {"channels": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
                               "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]},
}


def smoke_subset(items: list[tuple[str, str]], n: int, seed: int = 42) -> list[tuple[str, str]]:
    """Stratified (by T1/T2 label) random subset of participants for the smoke pipeline."""
    df = pd.read_parquet(PARTICIPANTS_FILE).set_index("participant_id")
    ids = np.array(sorted(p for p, _ in items))
    key = df.loc[ids, "T1_label"].astype(str) + "_" + df.loc[ids, "T2_label"].astype(str)
    rng = np.random.default_rng(seed)
    keep = []
    for _, grp in pd.Series(ids, index=ids).groupby(key.to_numpy()):
        k = max(1, int(round(n * len(grp) / len(ids))))
        keep += rng.choice(grp.to_numpy(), min(k, len(grp)), replace=False).tolist()
    keep = set(keep)
    return [(p, f) for p, f in items if p in keep]


def ensure_cache(dataset: str, cond: str, pre_cfg: dict | None = None, subset_n: int | None = None) -> None:
    base = dataset.split("@")[0]
    pre = deep_merge(pre_cfg or load_config("preprocess"), DATASET_VARIANTS.get(base, {}))
    if not cache_paths(dataset, cond)["meta"].exists():
        items = tdbrain_items(cond)
        if subset_n:
            items = smoke_subset(items, subset_n)
        log.info("Building epoch cache for %s/%s (%d recordings)", dataset, cond, len(items))
        run_dataset(items, pre, dataset, cond, pre.get("n_jobs", 1))


class TaskData:
    def __init__(self, cfg: dict, splits: dict | None = None):
        self.cfg = cfg
        self.task = cfg.get("task", "T1")
        self.dataset, self.cond = cfg.get("dataset", "tdbrain"), cfg.get("cond", "restEC")
        if cfg.get("subset_n"):  # smoke pipeline: separate cache "<dataset>@smoke<N>"
            self.dataset = f"{self.dataset}@smoke{cfg['subset_n']}"
        self.splits = splits or load_splits(SPLITS_FILE)
        self.labels: dict[str, int] = {k: int(v) for k, v in self.splits["tasks"][self.task]["labels"].items()}
        ensure_cache(self.dataset, self.cond, cfg.get("preprocess"), cfg.get("subset_n"))
        self._X = self._rows = self._feat = None
        self.max_epochs = cfg.get("max_epochs_per_person")
        # EEGNet may use fewer epochs per person than the feature track (compute budget); None = all kept epochs
        self.eeg_max_epochs = cfg["models"]["eegnet"].get("max_epochs_per_person") or self.max_epochs

    # --- raw epochs (track B)
    @property
    def X(self):
        if self._X is None:
            self._X, idx, self.meta = load_cache(self.dataset, self.cond)
            self._rows = participant_epochs(self._X, idx)
        return self._X

    @property
    def rows(self) -> dict:
        _ = self.X
        return self._rows

    # --- participant features (track A)
    def features(self) -> pd.DataFrame:
        if self._feat is None:
            f = feature_table(self.dataset, self.cond, self.cfg["models"]["logreg"], self.max_epochs)
            sel = self.cfg.get("features", "all")
            if sel == "faa_only":
                f = f[["FAA"]]
            elif sel == "no_faa":
                f = f.drop(columns=["FAA"], errors="ignore")
            if self.cfg.get("feature_transform") == "participant_relative":
                f = participant_relative(f)
            self._feat = f
        return self._feat

    def available(self) -> set[str]:
        return set(self.rows) if self.cfg.get("needs_epochs", True) else set(self.features().index)

    def fold(self, fold) -> dict:
        """Fold ids restricted to participants that survived preprocessing (exclusions are logged in QC).
        `fold` may be an outer-fold index, "final", or an explicit dict of ids (LiRA shadow models)."""
        fs = fold if isinstance(fold, dict) else get_fold(self.splits, self.task, fold)
        avail = set(self.features().index) & set(self.rows)
        out = {k: [p for p in fs[k] if p in avail] for k in ("train", "val") if k in fs}
        out["test"] = [p for p in fs.get("test", []) if p in avail]
        out["clients"] = {part: {k: [[p for p in c if p in avail] for c in cl] for k, cl in v.items()}
                          for part, v in fs["clients"].items()}
        return out

    def labels_for(self, fold_ids: dict, seed: int) -> dict[str, int]:
        """Labels as seen by training. E01 shuffles labels of train+val participants only (test stays real)."""
        lab = dict(self.labels)
        if self.cfg.get("shuffle_labels"):
            ids = sorted(fold_ids["train"] + fold_ids["val"])
            perm = np.random.default_rng(10_000 + seed).permutation([lab[p] for p in ids])
            lab.update(dict(zip(ids, perm.tolist(), strict=True)))
        return lab
