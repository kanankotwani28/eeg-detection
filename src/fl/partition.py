"""Client partitions come from artifacts/splits.json (IID or label-skew), never re-drawn here."""
from __future__ import annotations

from src.data.splits import get_clients
from src.train.torch_core import PData


def client_pdata(full: PData, fold_split: dict, partition: str, k: int) -> list[PData]:
    if k == 1:
        return [full.subset([p for p in full.ids if p in set(fold_split["train"])])]
    return [full.subset([p for p in ids if p in set(full.ids)]) for ids in get_clients(fold_split, partition, k)]
