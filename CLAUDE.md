# CLAUDE.md — Build spec: Federated, privacy-preserving EEG classification of MDD on TDBRAIN

Put this file in the root of an empty repository and start Claude Code there.
It is the complete brief. Build the project end to end, phase by phase, and
stop only at the checkpoints marked **STOP**.

---

## 0. What you are building

A reproducible research pipeline that answers one question:

> On TDBRAIN, under subject-level evaluation, how much accuracy do EEG-based MDD
> classifiers lose when trained federated and with participant-level
> differential privacy (DP)? How much privacy does that buy against a modern
> membership-inference attack? How do the models transfer to an independent
> dataset (Mumtaz)?

Components:

- **Two model tracks:**
  - (A) log band power + frontal alpha asymmetry → logistic regression.
  - (B) EEGNet-8,2 with GroupNorm.
- **Four training regimes:** centralised, local-only, federated (Flower FedAvg/FedProx),
  and federated + participant-level DP.
- **Evaluation:** per participant, with bootstrap CIs. Mumtaz is a one-time external test.
- **Privacy audit:** participant-level LiRA, reported as TPR at low FPR.
- **Dashboard:** a Streamlit dashboard showing all final results.

This is a research benchmark, **not a diagnostic tool**. Never write text, UI copy
or docstrings that claim clinical usefulness or call any EEG feature a
"biomarker of depression".

---

## 1. Non-negotiable rules

1. **No participant may appear in more than one of train / validation / test, or in more than one client.**
   Splits are made once, by participant, and saved to `artifacts/splits.json`. Every experiment reads that file.
   Never split by epoch, except in experiment E12, which is the deliberate leakage demonstration and must be labelled so.
2. **Nothing is fitted on test data.** Scalers, normalisers, thresholds, early stopping and hyperparameters are fitted on
   training data, or on an inner validation set of *different participants*.
3. **Mumtaz is not loaded by any training code.** It is read only in `src/eval/external.py` (phase 8). Log every access.
4. **No interpolation of missing channels.** Use the 19-channel intersection.
   Bad-channel interpolation *within* a recording is allowed. Log its count.
5. **No ComBat, no multi-agent framework, no LLM calls in the pipeline.** Plain deterministic Python modules.
6. **The privacy unit is the participant, not the epoch.** Report epoch-level ε only as a labelled comparison.
7. **Never commit data.** TDBRAIN's data-use agreement forbids redistribution. `data/`, `artifacts/cache/` and any
   EEG files go in `.gitignore`. The Docker image contains no data.
8. **Git attribution:** do NOT add `Co-Authored-By: Claude` or any Claude/Anthropic attribution line to commits or PRs.
   Do not list Claude in contributors, README credits, CITATION.cff or anywhere else. Commit as the repository owner only.
9. **Secrets** (Kaggle key, any tokens) live only in environment variables or `~/.kaggle/kaggle.json`.
   Never print them, log them or commit them. Add `.env` to `.gitignore`.
10. Every number shown in the dashboard or the report must be produced by code from logged runs. Never type results by hand.

---

## 2. Data sources and access

### 2.1 TDBRAIN (main dataset)

- Official source: Brainclinics. Login with ORCID, then accept the data-use agreement. https://brainclinics.com/?p=7420
- Descriptor: van Dijk et al., *Sci Data* 2022, https://www.nature.com/articles/s41597-022-01409-z
- Known facts:
  - 1,274 participants, 1,346 sessions.
  - Indications include MDD 426, ADHD 271, SMC 119, OCD 75.
  - 26 EEG channels plus non-EEG channels, 500 Hz.
  - 2 min eyes open (`restEO`) and 2 min eyes closed (`restEC`).
  - BIDS layout with BrainVision files.
  - ~30% of labels are blinded (`REPLICATION`, IDs beginning `sub-19`). `UNKNOWN` does **not** mean healthy.

Support **both** access modes through `configs/data.yaml`:

```yaml
tdbrain:
  mode: local            # local | kaggle
  local_path: /path/to/TDBRAIN    # folder containing participants.tsv and sub-*/
  kaggle_dataset: <owner>/<dataset-slug>   # only if mode: kaggle
```

- `mode: local`: verify `participants.tsv` exists and that at least one `sub-*/ses-*/eeg/*restEC*` file is readable.
- `mode: kaggle`: use the `kaggle` Python package (`kaggle datasets download -d <slug> --unzip -p data/tdbrain`).
  Credentials come from `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars, or `~/.kaggle/kaggle.json` (chmod 600).
  Only use a Kaggle copy that the user uploaded **privately** from their own licensed download.
  If the slug is a public third-party upload, **STOP** and tell the user it may breach the TDBRAIN agreement.
- The full archive is ~120 GB. Only first-session `restEC` (and `restEO` for E18) files plus `participants.tsv` are
  needed. If downloading, fetch only those where possible. Check free disk space before downloading.

### 2.2 Mumtaz (external test only)

- figshare: "MDD Patients and Healthy Controls EEG Data (New)", doi 10.6084/m9.figshare.4244171 (~0.9 GB, EDF).
  No credentials needed. Download with `requests` from the figshare API, or ask the user for the folder path.
- Expected:
  - 34 MDD and 30 healthy participants.
  - 19 EEG channels (10–20 names: T3/T4/T5/T6 = T7/T8/P7/P8 in TDBRAIN naming), 256 Hz.
  - Linked-ear reference.
  - Files for eyes closed (EC), eyes open (EO) and TASK. Some files are missing.
  - Files carry extra non-EEG channels: drop them.
- Parse labels from file names (e.g. `MDD S1 EC.edf`, `H S1 EC.edf`). Verify the counts and print a table.

### 2.3 Not used

ICMR and the "qEEG" folder are out of scope. Do not search the user's disk for them.

---

## 3. Environment

- Python 3.11. Use `uv` (or conda if uv is unavailable). Commit `pyproject.toml` + lock file.
- Core libraries:
  - `mne`, `mne-bids`, `numpy`, `scipy`, `pandas`, `pyarrow`
  - `scikit-learn`, `torch`
  - `flwr[simulation]`, `opacus`
  - `mlflow`, `hydra-core` (or plain YAML + pydantic)
  - `streamlit`, `plotly`
  - `pytest`, `ruff`, `kaggle`
  - optional: `autoreject`, `braindecode`
- Detect hardware. Everything must run on CPU. Use CUDA or Apple MPS if available. Never require a GPU.
- Set seeds everywhere through `src/utils/seed.py`: python, numpy, torch, cudnn deterministic.

---

## 4. Repository layout

```
.
├── CLAUDE.md                  # this file
├── README.md                  # how to reproduce, data access per DUA
├── pyproject.toml / uv.lock
├── Dockerfile                 # no data inside
├── Makefile                   # make setup | data | splits | preprocess | exp | report | dashboard | test | reproduce
├── configs/
│   ├── data.yaml  preprocess.yaml  models.yaml  fl.yaml  dp.yaml  audit.yaml
│   └── experiments/E01.yaml ... E21.yaml
├── src/
│   ├── data/      index.py  splits.py  tdbrain.py  mumtaz.py  download.py
│   ├── preprocess/ pipeline.py  qc.py
│   ├── features/  spectral.py
│   ├── models/    logreg.py  eegnet.py
│   ├── train/     central.py  local.py
│   ├── fl/        client.py  server.py  strategy.py  partition.py
│   ├── privacy/   participant_dp.py  accountant.py  lira.py
│   ├── eval/      metrics.py  bootstrap.py  external.py
│   ├── explain/   coefficients.py  eegnet_patterns.py  band_occlusion.py
│   ├── report/    tables.py  figures.py  export.py
│   └── utils/     seed.py  logging.py  hashing.py  paths.py
├── dashboard/app.py (+ pages/)
├── tests/
├── artifacts/   (gitignored: cache/, splits.json, results/)
└── data/        (gitignored)
```

---

## 5. Phases

After each phase:

1. Run `pytest`.
2. Commit with a clear message (no Claude attribution).
3. Print a short summary: what was produced and the key numbers.

### Phase 0: Data access and participant index

- `src/data/index.py` builds `artifacts/participants.parquet` with one row per participant:
  - ID, first-session ID, indication(s), age, sex.
  - Whether a `restEC` / `restEO` file exists, file paths.
  - BDI where available.
- Rules:
  - First session only.
  - Drop `REPLICATION` and `UNKNOWN` from the labelled pool, but keep a count.
- Define two tasks:
  - **T1:** MDD vs HEALTHY.
  - **T2:** MDD vs other diagnosed patients (ADHD, OCD, SMC, …), excluding healthy.
  - Use the primary indication. Store the comorbidity flag.
- Print and save `artifacts/results/cohort.json`:
  - Counts per task and class.
  - Age and sex per class.
  - Channel list of one `restEC` file.
- Confirm that all 19 Mumtaz positions exist in TDBRAIN:
  Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2.
- **STOP.** Show the user the counts and ask them to confirm:
  - (a) the T1/T2 label definitions;
  - (b) the adult age range (default 18–65);
  - (c) whether to exclude comorbid MDD.

  If HEALTHY with `restEC` < 50, propose making T2 the primary task.

### Phase 1: Splits

- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by participant and stratified by label.
  Add an age band to the stratification key if the classes allow.
- Inside each outer-train fold:
  - Make an inner 80/20 group split for validation.
  - Assign training participants to K clients (default K=4) in two variants:
    - **IID** (random).
    - **label-skew** (Dirichlet α=0.5 over class proportions, seed-fixed, with ≥5 participants per class per client
      where possible).
- Save `artifacts/splits.json` with a SHA-256 hash.
- Tests: no participant overlap anywhere; every participant appears in exactly one test fold.

### Phase 2: Preprocessing (per recording; no cross-subject statistics)

Defaults in `configs/preprocess.yaml`:

```yaml
channels: [Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2]
rename: {T3: T7, T4: T8, T5: P7, T6: P8}
bandpass: [1.0, 40.0]        # FIR, zero-phase
notch: 50.0
reference: average           # over the 19 channels, after picking
resample: 256
epoch_len_s: 4.0
epoch_overlap: 0.0
reject_ptp_uv: 150           # (E) expert to confirm
flat_uv: 1.0
min_epochs_per_person: 10    # (E)
max_epochs_per_person: 30
```

- Order of steps:
  1. Load.
  2. Drop non-EEG channels.
  3. Rename.
  4. Pick the 19 channels in fixed order.
  5. Units → µV.
  6. Filter.
  7. Notch.
  8. Average reference.
  9. Resample.
  10. Epoch.
  11. Reject.
  12. Cap (keep the first N clean epochs).
- Output, cached in `artifacts/cache/`:
  - `epochs_{dataset}_{cond}.npy` with shape (N_epochs, 19, 1024), float32.
  - `epoch_index.parquet`: participant, epoch number, rejected flag.
- QC report `artifacts/results/qc.json`:
  - Epochs kept per person.
  - Rejection rates.
  - Excluded participants.
  - A PSD sanity check: mean eyes-closed alpha (8–13 Hz) power at O1/O2 must exceed alpha at Fp1/Fp2 for most
    participants. Warn if not.
- Same pipeline for Mumtaz, but run only in phase 8.

### Phase 3: Feature baseline (track A)

- Welch PSD (2 s windows, 50% overlap) per epoch.
- Log absolute power in delta 1–4, theta 4–8, alpha 8–13, beta 13–30 Hz → 76 features.
- Add FAA = ln(alpha F4) − ln(alpha F3).
- Average per participant.
- Model: `StandardScaler` (train only) → `LogisticRegression(penalty="l2", class_weight="balanced")`.
  Tune C ∈ {0.01, 0.1, 1, 10} on the inner validation set.
- Gamma is excluded by default (muscle contamination near the 40 Hz edge).

### Phase 4: EEGNet (track B), centralised and local-only

- EEGNet-8,2 (F1=8, D=2, F2=16), temporal kernel 128 (fs/2), max-norm constraints as in Lawhern et al. 2018.
  **GroupNorm** instead of BatchNorm. Dropout 0.5. Input (B, 1, 19, 1024) → 2 logits.
- Per-recording, per-channel z-score of the input (statistics from that recording only).
- Training:
  - Adam, lr 1e-3, batch 64.
  - Weighted cross-entropy.
  - Max 100 epochs; early stopping on inner-validation participant-level balanced accuracy (patience 15).
- Per-person probability = mean of epoch softmax probabilities.
- Decision threshold: tuned on inner validation for balanced accuracy. AUC is also reported, threshold-free.
- **Local-only:** train on a single client's participants and evaluate on the same outer test fold. Report mean and
  worst across clients.
- Seeds: 3 per configuration (configurable).

### Phase 5: Federated (Flower simulation)

- `ClientApp`/`ServerApp` with Flower's simulation engine. A client = one client partition of the outer-train fold.
- Strategy: FedAvg, weighted by client sample size.
  Options: FedProx (μ ∈ {0.01, 0.1}); FedBN (keep norm layers local, for E8).
- Defaults:
  - 1 local epoch.
  - 60 rounds.
  - fraction_fit 1.0.
  - Server-side evaluation on the inner validation participants each round.
  - Keep the best round's global model.
- Logistic regression is also trained federated (FedAvg on weights). Use SGD with log-loss locally if simpler.
- Dropout test: each client is unavailable in a round with p=0.2 (seeded).
- Test: with K=1, the federated result must match centralised within noise.

### Phase 6: Participant-level differential privacy

Implement in `src/privacy/participant_dp.py`. Opacus's default PrivacyEngine accounts per sample (= per epoch).
That is **not** what we want, so do this instead:

- At each local step, sample each participant of the client independently with probability q (Poisson sampling).
- For each sampled participant, compute the gradient of the loss on one randomly chosen clean epoch.
  Optionally use the mean over all of their epochs (configurable).
- Clip each participant's gradient to L2 norm C (default 1.0).
- Sum the clipped gradients, add Gaussian noise N(0, σ²C²I), and divide by the expected batch size.
- Use per-sample-gradient machinery (Opacus `GradSampleModule` or `torch.func`) only to obtain per-participant
  gradients.
- Accounting:
  - Use the Opacus RDP accountant with `sample_rate=q`, `noise_multiplier=σ` and steps = total local steps
    across all rounds for that client.
  - Each participant belongs to one client, so the reported ε is the participant-level guarantee for that client's
    released updates. Report the max over clients.
- Sweep σ so that ε ∈ {1, 3, 8} at δ = 1e-5. ε = ∞ is the non-DP reference.
  Also compute, and label clearly, the epoch-level ε that standard Opacus would report.
- DP logistic regression: the same mechanism applied to the linear model.
- Tests:
  - The accountant reproduces a published example (e.g. Opacus docs).
  - Clipping bounds each participant's contribution (unit test with synthetic gradients).

### Phase 7: Privacy audit (LiRA, participant level)

- Threat model, to document in the report:
  - An honest-but-curious server, or anyone with query access to the released model.
  - Goal: infer whether a participant was in training.
- Pool: all labelled participants of task T1, or of the primary task.
- Train S shadow models (S=32 for logistic regression, S=16 for EEGNet; configurable), each on a random 50% of the
  pool, using the same recipe as the target.
- Score per participant: mean over their epochs of the logit of the correct class (logit-scaled confidence).
- For each target participant:
  - Fit Gaussians to the in- and out-shadow scores.
  - Compute the likelihood ratio on the target model's score.
- Report:
  - Full ROC and AUC.
  - TPR at FPR = 1% and 5%.
  - TPR at 0.1% only if there are ≥1,000 non-member evaluations; otherwise state that it cannot be estimated.
- Run for: logistic regression and EEGNet, non-DP vs each DP level, centralised and federated.
- Sanity check: a target model trained without any of the evaluated participants gives AUC ≈ 0.5.

### Phase 8: External test on Mumtaz

- Load Mumtaz (eyes closed) with the identical preprocessing config. Verify the 19-channel order.
- Models used:
  - The final model of each configuration, retrained on all TDBRAIN T1 participants with the hyperparameters chosen
    in CV.
  - The threshold fixed from TDBRAIN validation.
- Evaluate **once**. Report per-person balanced accuracy, AUC and sensitivity/specificity with 2,000-sample bootstrap
  CIs.
- Also run E13, the dataset-identity probe: logistic regression on band-power features predicting dataset
  (TDBRAIN-HC vs Mumtaz-HC), with group CV. Report AUC.
- Also run E20: within-Mumtaz subject-level vs segment-level CV.

### Phase 9: Experiments, report and packaging

Run every experiment in the matrix below through `make exp E=E04` and `make exp-all`.

- Log each run to MLflow (local `./mlruns`), one run per (experiment ID, fold, seed). Log:
  - The config, the config hash, the split hash, the `participants.tsv` hash, the git commit.
  - Seeds.
  - Metrics, confusion matrix, ε, the model checkpoint and per-person predictions (parquet).
- `src/report/export.py` aggregates the MLflow runs into `artifacts/results/*.parquet` + `summary.json`.
  The dashboard reads only these files.
- Statistics:
  - Mean ± SD across folds and seeds.
  - Participant-level bootstrap 95% CIs (2,000 resamples).
  - Paired bootstrap of the difference for each comparison.
  - Do **not** report p-values from 5-fold Wilcoxon tests as significant.

| ID | Question | Model | Data / task | Regime | Priority |
|---|---|---|---|---|---|
| E01 | Shuffled-label sanity check → ~50% | LR, EEGNet | T1 | Centralised | Essential |
| E02 | Spectral baseline | LR | T1 | Centralised | Essential |
| E03 | Harder contrast | LR | T2 | Centralised | Essential |
| E04 | Deep vs features | EEGNet | T1 (+T2) | Centralised | Essential |
| E05 | Single-site loss | LR, EEGNet | T1 | Local only | Essential |
| E06 | FL cost, IID vs label-skew | LR, EEGNet | T1 | FedAvg | Essential |
| E07 | FedProx under skew | EEGNet | T1 | FedProx | Optional |
| E08 | Local normalisation | EEGNet | T1 | FedBN | Optional |
| E09 | Privacy cost, ε ∈ {1,3,8,∞} | LR, EEGNet | T1 | FedAvg + DP | Essential |
| E10 | LiRA audit, DP vs non-DP | LR, EEGNet | T1 | All | Essential |
| E11 | External transfer | Best LR/EEGNet | TDBRAIN → Mumtaz | Frozen | Essential |
| E12 | Segment-split leakage demo (deliberately wrong) | EEGNet | T1 | Centralised | Essential |
| E13 | Dataset-identity probe | LR | TDBRAIN-HC vs Mumtaz-HC | Centralised | Essential |
| E14 | 26-ch interpolation ablation | LR, EEGNet | T1 | Centralised | Optional |
| E15 | Preprocessing sensitivity (ICA on/off, reference) | LR | T1 | Centralised | Optional |
| E16 | FAA alone | LR | T1 | Centralised | Essential |
| E17 | Number of clients K ∈ {2,4,8} | EEGNet | T1 | FedAvg | Optional |
| E18 | Eyes-open sensitivity | LR | T1 (EO) | Centralised | Optional |
| E19 | 1 min vs 2 min of data | LR | T1 | Centralised | Optional |
| E20 | Mumtaz subject vs segment CV | LR, EEGNet | Mumtaz | Centralised | Optional |
| E21 | Export predictions for the TDBRAIN blinded diagnostic validation | Best | Held-out sample | Frozen | Optional (needs the user) |

Run order: E01 → E02 → E16 → E03 → E04 → E05 → E06 → E09 → E10 → E12 → E11 → E13, then the optional ones if time
allows. If E01 is not ≈ 50%, **STOP** and find the leak before anything else.

Explainability, saved for the dashboard:

- **Logistic regression:** standardised coefficients and permutation importance, grouped by band × region. Keep only
  features whose sign is stable in ≥4/5 folds.
- **EEGNet:** spatial patterns of the depthwise filters as topomaps (MNE `plot_topomap` on the 19-channel montage),
  plus band-occlusion (remove one band from the input, measure the drop in balanced accuracy).
- Label everything "explains the model, not the disorder".

Packaging:

- **README:** overview, data access for TDBRAIN (ORCID + DUA) and Mumtaz, the Kaggle option, `make reproduce`,
  hardware notes, limitations, citations.
- **Dockerfile:** builds the environment; data mounted at runtime.
- **CITATION.cff:** the repository owner as author. Cite TDBRAIN, Mumtaz, EEGNet, Flower and Opacus in the README.
- `make reproduce` runs, on a small-subset smoke config (`configs/experiments/smoke.yaml`): splits → preprocess →
  E01, E02, E04, E06, E09 → export → dashboard data. It finishes in under 30 min on CPU.

### Phase 10: Dashboard (Streamlit)

`make dashboard` runs `streamlit run dashboard/app.py`. It is read-only, local, needs no credentials, and has no
data upload.

Pages:

1. **Overview:**
   - The study question.
   - The "not a diagnostic tool" banner.
   - A cohort flowchart: 1,274 → first session → `restEC` → labels → final N per task.
   - Dataset table.
2. **Cohort and QC:**
   - Class counts.
   - Age/sex distributions per class.
   - Epochs kept per person.
   - Rejection rates.
   - Group-mean PSD per class.
3. **Model performance:**
   - Balanced accuracy and AUC (with CIs) for every model × regime × task, as a grouped bar or dot plot with error
     bars.
   - ROC curves.
   - Calibration plots.
   - Confusion matrices.
   - A filter by experiment ID.
4. **Federated learning:**
   - Round-by-round validation curves.
   - IID vs label-skew.
   - FedAvg vs FedProx vs FedBN.
   - K-clients plot.
   - Per-client and worst-client results.
5. **Privacy:**
   - Accuracy vs ε curve: the headline figure.
   - The LiRA ROC on log-log axes.
   - TPR at 1% / 5% FPR, DP vs non-DP.
   - An explicit note that participant-level and epoch-level ε differ.
6. **External test:**
   - TDBRAIN CV vs Mumtaz scores, with CIs and the gap.
   - Dataset-identity probe AUC.
   - Segment-split leakage demo (E12) vs subject split.
7. **Explainability:**
   - Stable LR coefficients by band × region.
   - EEGNet topomaps.
   - Band-occlusion bars.
8. **Reproducibility:**
   - Git commit.
   - Config and split hashes.
   - Seeds.
   - Package versions.
   - Links to MLflow runs.

Use Plotly. Every chart shows its n (participants) and its CI. No hard-coded numbers.

---

## 6. Tests (`pytest`, must pass before each commit)

- `test_splits.py`: no participant overlap across train/val/test/clients; each participant in exactly one test fold;
  the split hash is stable.
- `test_preprocess.py`: output shape (n, 19, 1024); channel order; units in µV; sfreq 256; no NaNs;
  T3→T7 renaming works.
- `test_no_test_fitting.py`: scalers and thresholds receive only training/validation indices
  (mock and assert on the indices).
- `test_dp.py`: per-participant clipping bound; noise scale; the accountant matches a reference value.
- `test_fl.py`: K=1 federated ≈ centralised on a tiny synthetic dataset.
- `test_lira.py`: on synthetic data where members are trivially separable, AUC > 0.9; with random labels, AUC ≈ 0.5.
- `test_mumtaz_isolation.py`: grep or import-graph check that only `src/eval/external.py` reads Mumtaz paths.

---

## 7. Checkpoints where you must STOP and ask the user

1. After phase 0: confirm the label definitions, age range, comorbidity rule and primary task (show the counts).
2. Before any download larger than 5 GB: confirm the disk space and the path.
3. If the Kaggle slug looks like a public third-party copy of TDBRAIN.
4. If E01 (shuffled labels) is not ≈ 50%.
5. If preprocessing rejects more than 30% of participants: ask the user to confirm the thresholds
   (`reject_ptp_uv`, `min_epochs_per_person`) with their EEG expert.
6. Before pushing to any remote repository.

Everything else: decide sensibly, record the decision in `DECISIONS.md` (date, decision, reason), and continue.

---

## 8. Definition of done

- `make test` passes. `make reproduce` runs on a fresh clone with data paths set.
- All essential experiments (E01–E06, E09–E13, E16) are logged in MLflow, with results exported to
  `artifacts/results/`.
- The dashboard shows all eight pages, populated from the exported results.
- `README.md`, `DECISIONS.md`, `CITATION.cff` and `Dockerfile` exist. No data or secrets are in git history.
- `REPORT.md` is auto-generated from the results. It contains:
  - The cohort table.
  - The main results table with CIs.
  - The privacy-utility figure.
  - The external-test table.
  - The limitations: simulated clients ≠ hospitals; small external set; medication not tracked; label noise;
    not a diagnostic tool.
