# Federated, participant-level DP EEG classification of MDD on TDBRAIN — research benchmark, not a diagnostic tool.
PY ?= uv run python
E ?= E02
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: setup data splits preprocess exp exp-all exp-optional report dashboard test lint reproduce synthetic clean-cache

setup:            ## create the environment from the lock file
	uv sync --frozen

data:             ## (kaggle mode: download) + build artifacts/participants.parquet and cohort.json  -> STOP and review
	$(PY) -m src.data.download
	$(PY) -m src.data.index

splits:           ## frozen participant-level splits -> artifacts/splits.json (never re-drawn; --force to rebuild)
	$(PY) -m src.data.splits

preprocess:       ## per-recording preprocessing -> artifacts/cache/, QC -> artifacts/results/qc.json
	$(PY) -m src.preprocess.pipeline --cond restEC

exp:              ## one experiment: make exp E=E04
	$(PY) -m src.experiments $(E)

exp-all:          ## essential experiments in the prescribed order (stops if E01 is not ~50%)
	$(PY) -m src.experiments essential

exp-optional:
	$(PY) -m src.experiments E07 E08 E14 E15 E17 E18 E19 E20

report:           ## MLflow -> artifacts/results/*.parquet + summary.json, explainability, REPORT.md
	$(PY) -m src.report.export
	$(PY) -m src.report.tables

dashboard:        ## read-only local dashboard
	uv run streamlit run dashboard/app.py

test:
	$(PY) -m pytest -p no:warnings

lint:
	uv run ruff check src tests dashboard

synthetic:        ## generate SYNTHETIC test data under data/synthetic (for trying the pipeline without TDBRAIN)
	$(PY) -m src.data.synthetic --tdbrain data/synthetic/tdbrain --mumtaz data/synthetic/mumtaz

reproduce:        ## smoke pipeline (< 30 min CPU): splits -> preprocess subset -> E01 E02 E04 E06 E09 -> export -> report
	$(PY) -m src.data.index
	$(PY) -m src.data.splits
	$(PY) -m src.experiments E01 E02 E04 E06 E09 --smoke
	$(PY) -m src.report.export
	$(PY) -m src.report.tables

clean-cache:
	rm -rf artifacts/cache

exp-budget:       ## real-data essential pipeline with the CPU budget overlay (index -> ... -> E13, report after each)
	PY="$(PY)" scripts/run_essential.sh budget
