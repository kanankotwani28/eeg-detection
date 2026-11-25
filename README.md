# Federated, privacy-preserving EEG classification of MDD on TDBRAIN

> **Research benchmark — not a diagnostic tool.** Nothing in this repository is a clinical test, and no EEG
> feature here is treated as a biomarker of depression.

This pipeline answers one question:

> On TDBRAIN, under subject-level evaluation, how much accuracy do EEG-based MDD classifiers lose when trained
> federated and with participant-level differential privacy (DP)? How much privacy does that buy against a modern
> membership-inference attack (LiRA)? How do the models transfer to an independent dataset (Mumtaz)?

It is built from the spec in [`CLAUDE.md`](CLAUDE.md) and the review in
[`docs/research_architecture_plan.md`](docs/research_architecture_plan.md). Decisions the spec left open are in
[`DECISIONS.md`](DECISIONS.md).

| Component | What |
|---|---|
| Model tracks | (A) log band power (δ θ α β × 19 ch) + frontal alpha asymmetry → L2 logistic regression; (B) EEGNet-8,2 with GroupNorm |
| Regimes | centralised · local-only (per client) · federated (Flower FedAvg / FedProx / FedBN) · federated + **participant-level** DP |
| Evaluation | stratified group 5-fold CV by participant (frozen in `artifacts/splits.json`), participant-level metrics, 2,000-resample bootstrap CIs, paired bootstrap differences |
| Privacy audit | participant-level LiRA, TPR at 1% / 5% FPR (0.1% only with ≥ 1,000 non-member evaluations) |
| External test | frozen models evaluated **once** on Mumtaz (eyes closed) |
| Dashboard | Streamlit, 8 pages, reads only exported results |

## Data access

**Never commit data.** `data/`, `artifacts/` and all EEG file types are git-ignored, and the Docker image contains no
data.

### TDBRAIN (main dataset)
1. Log in with ORCID at Brainclinics and accept the data-use agreement: <https://brainclinics.com/?p=7420>.
2. Download the archive (≈ 120 GB). Only `participants.tsv` and first-session `restEC` files are required
   (`restEO` only for E18).
3. Point the pipeline at it, either in `configs/data.yaml` (`tdbrain.local_path`) or with
   `export TDBRAIN_PATH=/path/to/TDBRAIN` (the folder that holds `participants.tsv` and `sub-*/`).

**Kaggle option.** Use this only for a copy that **you** uploaded **privately** from your own licensed download. Set
`tdbrain.mode: kaggle`, `kaggle_dataset: <you>/<slug>` and `kaggle_private_confirmed: true`. Credentials come from
`KAGGLE_USERNAME` + `KAGGLE_KEY` or `~/.kaggle/kaggle.json` (chmod 600). They are never printed or logged. The
download refuses any dataset not owned by your Kaggle account, because a public third-party copy may breach the
TDBRAIN agreement.

### Mumtaz (external test only)
"MDD Patients and Healthy Controls EEG Data (New)", figshare doi
[10.6084/m9.figshare.4244171](https://doi.org/10.6084/m9.figshare.4244171) (≈ 0.9 GB, EDF, no login). It is read
only by `src/eval/external.py`, which downloads it on first use if `data/mumtaz` (or `MUMTAZ_PATH`) is empty. Every
access is logged to `artifacts/results/mumtaz_access.log`.

## Reproduce

```bash
make setup                  # uv sync --frozen (Python 3.11)
make data                   # participant index + cohort.json -> review the counts (phase-0 checkpoint)
make splits                 # frozen participant-level splits
make preprocess             # epochs cache + QC
make exp E=E01              # must be ~50% before anything else
make exp-all                # essential experiments in order: E01 E02 E16 E03 E04 E05 E06 E09 E10 E12 E11 E13
make report                 # MLflow -> artifacts/results/*.parquet, explainability, REPORT.md
make dashboard              # http://localhost:8501
```

`make reproduce` runs the smoke configuration (`configs/experiments/smoke.yaml`: a 120-participant stratified
subset, 2 folds, 1 seed, short training). It covers splits → preprocessing → E01, E02, E04, E06, E09 → export →
report, and is designed to finish in under 30 minutes on CPU.

To try the pipeline without TDBRAIN, generate **synthetic** data in both layouts with `make synthetic`, then set
`TDBRAIN_PATH=data/synthetic/tdbrain MUMTAZ_PATH=data/synthetic/mumtaz`. The report and dashboard flag synthetic
results in red.

```bash
docker build -t eeg-fl-dp .
docker run --rm -v /path/to/TDBRAIN:/data/tdbrain:ro -v $PWD/artifacts:/app/artifacts -v $PWD/mlruns:/app/mlruns eeg-fl-dp make reproduce
```

## Layout

```
configs/            data, preprocess, models, fl, dp, audit + experiments/E01..E21.yaml, smoke.yaml
src/data            participant index, frozen splits, TDBRAIN loader, Mumtaz parser, downloads, synthetic generator
src/preprocess      per-recording pipeline (19 ch, 1-40 Hz, notch, CAR, 256 Hz, 4 s epochs, reject, cap) + QC
src/features        Welch band power + FAA
src/models          logistic regression, EEGNet-8,2 (GroupNorm, max-norm)
src/train           centralised, local-only, shared torch core
src/fl              Flower client/server/strategy, partitions, federated regime
src/privacy         participant-level DP-SGD, RDP accounting (Opacus), LiRA
src/eval            metrics, bootstrap, external test (only Mumtaz reader), blinded export (E21)
src/explain         LR coefficients/permutation importance, EEGNet spatial patterns, band occlusion
src/report          MLflow export, tables, figures, REPORT.md
src/experiments.py  experiment runner (one MLflow run per experiment x variant x model x fold x seed)
dashboard/          Streamlit app (8 pages)
tests/              pytest suite (synthetic data only)
```

## Hardware

Everything runs on CPU. CUDA or Apple MPS are used automatically when available, but a GPU is never required.
EEGNet has about 3,000 parameters. A full 5-fold × 3-seed EEGNet configuration on about 700 participants takes
hours on a laptop CPU. The LiRA audit for EEGNet (16 shadow models per DP level) is the most expensive step;
reduce `configs/audit.yaml` if needed.

## Limitations

- Simulated clients are random partitions of one archive, **not hospitals**.
- The external set (Mumtaz, 64 people) is small, so its CIs are wide.
- Medication was not tracked systematically in TDBRAIN.
- Diagnostic labels from clinical practice are noisy.
- ε is per client release and per participant. The epoch-level ε is shown only for comparison and is not a
  participant guarantee.
- Explainability outputs explain the model, not the disorder.
- Not a diagnostic tool.

## Citations

- van Dijk, H. et al. (2022). The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database. *Scientific Data* 9, 333. <https://doi.org/10.1038/s41597-022-01409-z>
- Mumtaz, W. et al. (2017). A wavelet-based technique to predict treatment outcome for major depressive disorder. *PLoS ONE* 12(2): e0171409; data: Mumtaz, W. (2016). MDD Patients and Healthy Controls EEG Data (New). figshare. <https://doi.org/10.6084/m9.figshare.4244171>
- Lawhern, V. J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *J. Neural Eng.* 15, 056013.
- Beutel, D. J. et al. (2020). Flower: A friendly federated learning research framework. arXiv:2007.14390.
- Yousefpour, A. et al. (2021). Opacus: User-friendly differential privacy library in PyTorch. arXiv:2109.12298.
- McMahan, H. B. et al. (2018). Learning differentially private recurrent language models (user-level DP). ICLR.
- Carlini, N. et al. (2022). Membership inference attacks from first principles (LiRA). IEEE S&P.
- Brookshire, G. et al. (2024). Data leakage in deep learning studies of translational EEG. *Front. Neurosci.* 18:1373515.

See [`CITATION.cff`](CITATION.cff) to cite this repository.
