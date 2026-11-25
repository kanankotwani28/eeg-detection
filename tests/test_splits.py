import json

import numpy as np
import pandas as pd
import pytest

from src.data.splits import K_VARIANTS, load_splits, make_splits


def fake_participants(n=240, seed=0):
    r = np.random.default_rng(seed)
    lab = r.choice(["MDD", "HEALTHY", "ADHD", "OCD"], n, p=[0.4, 0.3, 0.2, 0.1])
    return pd.DataFrame({"participant_id": [f"sub-{i:08d}" for i in range(n)], "age": r.uniform(18, 65, n),
                         "T1_label": np.where(lab == "MDD", 1, np.where(lab == "HEALTHY", 0, -1)),
                         "T2_label": np.where(lab == "MDD", 1, np.where(lab != "HEALTHY", 0, -1))})


@pytest.fixture(scope="module")
def splits():
    return make_splits(fake_participants())


def test_no_overlap_anywhere(splits):
    for task, t in splits["tasks"].items():
        for f in t["folds"]:
            tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
            assert not (tr & va) and not (tr & te) and not (va & te), (task, f["fold"])
            for part in ("iid", "label_skew"):
                for k, clients in f["clients"][part].items():
                    allc = [p for c in clients for p in c]
                    assert len(allc) == len(set(allc)), "participant in more than one client"
                    assert set(allc) == tr, "clients must partition exactly the training participants"
                    assert not (set(allc) & (va | te))
                    assert len(clients) == int(k)


def test_each_participant_in_exactly_one_test_fold(splits):
    for t in splits["tasks"].values():
        tests = [p for f in t["folds"] for p in f["test"]]
        assert len(tests) == len(set(tests)) == t["n"]
        assert set(tests) == set(t["labels"])


def test_both_classes_in_every_fold_and_label_skew_min(splits):
    t = splits["tasks"]["T1"]
    for f in t["folds"]:
        assert {t["labels"][p] for p in f["test"]} == {0, 1}
        assert {t["labels"][p] for p in f["val"]} == {0, 1}
        for c in f["clients"]["label_skew"]["4"]:
            ys = [t["labels"][p] for p in c]
            # >= 5 per class per client where possible
            assert min(ys.count(0), ys.count(1)) >= min(5, len(f["train"]) // 8)
    assert set(map(str, K_VARIANTS)) >= set(t["folds"][0]["clients"]["iid"])


def test_split_hash_stable_and_tamper_detected(tmp_path):
    a, b = make_splits(fake_participants()), make_splits(fake_participants())
    assert a["sha256"] == b["sha256"]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(a))
    assert load_splits(p)["sha256"] == a["sha256"]
    a["tasks"]["T1"]["folds"][0]["test"].append("sub-intruder")
    p.write_text(json.dumps(a))
    with pytest.raises(ValueError):
        load_splits(p)
