"""Fast real-data pass: two parallel streams. `python scripts/run_fast.py lr` | `python scripts/run_fast.py eegnet`.
LR stream runs at the full specification; the EEGNet stream uses the `fast` overlay and fewer variants.
Each stream checks E01 for its model first and stops if it is not ~50%."""
import json
import subprocess
import sys

sys.path.insert(0, ".")
from src.experiments import check_e01, run_experiment  # noqa: E402

STREAMS = {
    "lr": [("E01", {}), ("E02", {}), ("E16", {}), ("E03", {}), ("E05", {}), ("E06", {}), ("E09", {}), ("E10", {}),
           ("E13", None)],
    "eegnet": [("E01", {"models": {"train": {"folds": [0, 1]}}}), ("E04", {}),
               ("E12", {"models": {"train": {"folds": [0, 1]}}}),
               ("E06", {"grid": [{"variant": "iid", "fl": {"partition": "iid"}},
                                 {"variant": "label_skew", "fl": {"partition": "label_skew"}}]}),
               ("E09", {}), ("E05", {"grid": [{"variant": "iid", "fl": {"partition": "iid"}}]}),
               ("E10", {"audit_regimes": ["federated"]})],
}
MODEL = {"lr": "logreg", "eegnet": "eegnet"}


def export():
    subprocess.run([sys.executable, "-m", "src.report.export"], check=False)
    subprocess.run([sys.executable, "-m", "src.report.tables"], check=False)


def main(stream):
    overlay = "fast" if stream == "eegnet" else None
    for exp, over in STREAMS[stream]:
        print(f"=== {stream} {exp} ===", flush=True)
        o = dict(over or {})
        if over is not None:
            o["run_models"] = [MODEL[stream]]
        res = run_experiment(exp, overrides=o, overlay=overlay)
        if exp == "E01":
            chk = check_e01(res)
            print(json.dumps({"E01_check": chk}), flush=True)
            if not all(v["passes"] for v in chk.values()):
                raise SystemExit("STOP: E01 shuffled-label check is not ~50%")
        export()
    print(f"=== {stream} DONE ===", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
