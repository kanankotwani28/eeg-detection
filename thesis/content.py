"""Thesis content, written once and rendered to LaTeX and Word by thesis/render.py.
Every number is read from thesis/values.json (built from the logged runs by build_figures.py).

Block types:
  ("chapter", title) ("section", title) ("subsection", title) ("p", text) ("fig", key, caption, width)
  ("table", key, caption, header, rows, colspec) ("list", ordered, items) ("deflist", items) ("eq", latex, text)
  ("pagebreak",)
Inline: **bold**, *italic*, [@key] citations, ⟦fig:key⟧ ⟦tab:key⟧ cross-references.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = {
    "title": "PRIVACY-PRESERVING FEDERATED LEARNING FOR EEG-BASED CLASSIFICATION OF MAJOR DEPRESSIVE DISORDER",
    "title_case": "Privacy-Preserving Federated Learning for EEG-Based Classification of Major Depressive Disorder",
    "subtitle": "A Leakage-Safe Benchmark on TDBRAIN",
    "student": "KANAN KOTWANI",
    "student_case": "Kanan Kotwani",
    "roll": "BTECH/15140/24",
    "guide": "Dr. Gyanendra Tiwary",
    "hod": "Dr. Ravi Shankar Pandey",
    "degree": "BACHELOR OF TECHNOLOGY",
    "branch": "COMPUTER SCIENCE AND ENGINEERING",
    "spec": "(ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)",
    "branch_case": "Computer Science and Engineering (Artificial Intelligence and Machine Learning)",
    "dept": "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING",
    "inst": "BIRLA INSTITUTE OF TECHNOLOGY MESRA",
    "campus": "OFF-CAMPUS PATNA – 800014",
    "semester": "Monsoon Semester 2026 (PROJECT-I, AI24300)",
}

REFS = {
    "who": 'World Health Organization, "Depressive disorder (depression)," Fact sheet, 2023. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/depression',
    "tdbrain": 'H. van Dijk *et al.*, "The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database," *Scientific Data*, vol. 9, art. 333, 2022. doi: 10.1038/s41597-022-01409-z.',
    "mumtaz17": 'W. Mumtaz, L. Xia, S. S. A. Ali, M. A. M. Yasin, M. Hussain and A. S. Malik, "Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD)," *Biomedical Signal Processing and Control*, vol. 31, pp. 108–115, 2017. doi: 10.1016/j.bspc.2016.07.006.',
    "mumtazdata": 'W. Mumtaz, "MDD patients and healthy controls EEG data (new)," figshare, 2016. doi: 10.6084/m9.figshare.4244171.',
    "review": 'D. de Aguiar Neto and J. L. G. Rosa, "Depression biomarkers using non-invasive EEG: A review," *Neuroscience & Biobehavioral Reviews*, vol. 105, pp. 83–93, 2019. doi: 10.1016/j.neubiorev.2019.07.021.',
    "faa": 'A. G. van der Vinne, F. Wiersma, M. Arns *et al.*, "Frontal alpha asymmetry as a diagnostic marker in depression: Fact or fiction? A meta-analysis," *NeuroImage: Clinical*, vol. 16, pp. 79–87, 2017. doi: 10.1016/j.nicl.2017.07.006.',
    "leak": 'G. Brookshire *et al.*, "Data leakage in deep learning studies of translational EEG," *Frontiers in Neuroscience*, vol. 18, art. 1373515, 2024. doi: 10.3389/fnins.2024.1373515.',
    "varo": 'G. Varoquaux, "Cross-validation failure: Small sample sizes lead to large error bars," *NeuroImage*, vol. 180, pp. 68–77, 2018. doi: 10.1016/j.neuroimage.2017.06.061.',
    "eegnet": 'V. J. Lawhern *et al.*, "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces," *Journal of Neural Engineering*, vol. 15, art. 056013, 2018. doi: 10.1088/1741-2552/aace8c.',
    "gn": 'Y. Wu and K. He, "Group normalization," in *Proc. European Conference on Computer Vision (ECCV)*, 2018, pp. 3–19.',
    "fedavg": 'B. McMahan, E. Moore, D. Ramage, S. Hampson and B. Agüera y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. 20th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR vol. 54, 2017, pp. 1273–1282.',
    "fedprox": 'T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar and V. Smith, "Federated optimization in heterogeneous networks," in *Proc. Machine Learning and Systems*, vol. 2, 2020, pp. 429–450.',
    "fedbn": 'X. Li, M. Jiang, X. Zhang, M. Kamp and Q. Dou, "FedBN: Federated learning on non-IID features via local batch normalization," in *Proc. International Conference on Learning Representations (ICLR)*, 2021.',
    "rieke": 'N. Rieke *et al.*, "The future of digital health with federated learning," *npj Digital Medicine*, vol. 3, art. 119, 2020. doi: 10.1038/s41746-020-00323-1.',
    "kaissis": 'G. A. Kaissis, M. R. Makowski, D. Rückert and R. F. Braren, "Secure, privacy-preserving and federated machine learning in medical imaging," *Nature Machine Intelligence*, vol. 2, pp. 305–311, 2020. doi: 10.1038/s42256-020-0186-1.',
    "dwork": 'C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," *Foundations and Trends in Theoretical Computer Science*, vol. 9, no. 3–4, pp. 211–407, 2014.',
    "abadi": 'M. Abadi *et al.*, "Deep learning with differential privacy," in *Proc. ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 2016, pp. 308–318. doi: 10.1145/2976749.2978318.',
    "userdp": 'H. B. McMahan, D. Ramage, K. Talwar and L. Zhang, "Learning differentially private recurrent language models," in *Proc. International Conference on Learning Representations (ICLR)*, 2018.',
    "rdp": 'I. Mironov, "Rényi differential privacy," in *Proc. IEEE 30th Computer Security Foundations Symposium (CSF)*, 2017, pp. 263–275. doi: 10.1109/CSF.2017.11.',
    "balle": 'B. Balle, G. Barthe, M. Gaboardi, J. Hsu and T. Sato, "Hypothesis testing interpretations and Rényi differential privacy," in *Proc. 23rd Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR vol. 108, 2020, pp. 2496–2506.',
    "dlg": 'L. Zhu, Z. Liu and S. Han, "Deep leakage from gradients," in *Advances in Neural Information Processing Systems*, vol. 32, 2019.',
    "shokri": 'R. Shokri, M. Stronati, C. Song and V. Shmatikov, "Membership inference attacks against machine learning models," in *Proc. IEEE Symposium on Security and Privacy*, 2017, pp. 3–18. doi: 10.1109/SP.2017.41.',
    "lira": 'N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis and F. Tramèr, "Membership inference attacks from first principles," in *Proc. IEEE Symposium on Security and Privacy*, 2022, pp. 1897–1914. doi: 10.1109/SP46214.2022.9833649.',
    "flower": 'D. J. Beutel *et al.*, "Flower: A friendly federated learning research framework," arXiv:2007.14390, 2020.',
    "opacus": 'A. Yousefpour *et al.*, "Opacus: User-friendly differential privacy library in PyTorch," arXiv:2109.12298, 2021.',
    "mne": 'A. Gramfort *et al.*, "MEG and EEG data analysis with MNE-Python," *Frontiers in Neuroscience*, vol. 7, art. 267, 2013. doi: 10.3389/fnins.2013.00267.',
    "welch": 'P. Welch, "The use of fast Fourier transform for the estimation of power spectra," *IEEE Trans. Audio and Electroacoustics*, vol. 15, no. 2, pp. 70–73, 1967. doi: 10.1109/TAU.1967.1161901.',
    "sklearn": 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.',
    "torch": 'A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems*, vol. 32, 2019.',
    "adam": 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. International Conference on Learning Representations (ICLR)*, 2015.',
    "mlflow": 'M. Zaharia *et al.*, "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Engineering Bulletin*, vol. 41, no. 4, pp. 39–45, 2018.',
    "boot": 'B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York: Chapman & Hall, 1993.',
}


def load():
    return json.loads((HERE / "values.json").read_text())


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def f2(x):
    return "—" if x is None else f"{x:.2f}"


def build():
    V = load()
    S = V["summary"]

    def row(exp, model="logreg", variant=None, eps="inf"):
        for r in S:
            if r["experiment"] == exp and r["model"] == model and r["eps"] == eps and (variant is None or r["variant"] == variant):
                return r
        raise KeyError((exp, model, variant, eps))

    def ci(r, m="balanced_accuracy"):
        return f"{f3(r[m])} [{f3(r[m + '_lo'])}, {f3(r[m + '_hi'])}]"

    def comp(label, metric="balanced_accuracy"):
        for c in V["comparisons"]:
            if c["comparison"] == label and c["metric"] == metric:
                return c
        raise KeyError(label)

    def lira(regime, eps, model="logreg"):
        for r in V["lira"]:
            if r["model"] == model and r["regime"] == regime and r["eps"] == eps:
                return r
        raise KeyError((regime, eps))

    def ext(source):
        for r in V["external"]:
            if r.get("source") == source:
                return r
        raise KeyError(source)

    probe = [r for r in V["external"] if r["kind"] == "probe"][0]
    co, qc, c = V["cohort"], V["qc"], V["c"]
    t1m, t1h = co["T1"]["MDD"], co["T1"]["HEALTHY"]
    t2o = co["T2"]["OTHER"]
    e01, e01n = row("E01"), row("E01", "eegnet")
    e02, e03, e16, e22 = row("E02"), row("E03"), row("E16"), row("E22")
    e04 = row("E04", "eegnet", "T1")
    e05i, e05s = row("E05", variant="iid"), row("E05", variant="label_skew")
    e06i, e06s, e06d = row("E06", variant="iid"), row("E06", variant="label_skew"), row("E06", variant="iid_dropout")
    e09 = {e: row("E09", eps=e) for e in ("1", "3", "8", "inf")}
    L = {k: lira("federated", k) for k in ("1", "3", "8", "inf")}
    Lc = lira("central", "inf")
    x_c, x_f, x_d = ext("LR-central"), ext("LR-FedAvg"), ext("LR-FedAvg-DP-eps8")
    c_age = comp("Band power LR - age-only LR")
    c_faa = comp("FAA only - full LR")
    c_fl = comp("LR FedAvg IID - LR central")
    c_fla = comp("LR FedAvg IID - LR central", "auc")
    c_sk = comp("LR FedAvg label-skew - IID")
    c_dp = {e: comp(f"logreg DP eps={e} - non-DP") for e in ("1", "3", "8")}
    c_eeg = comp("EEGNet - LR (central, T1)")
    n = int(e02["n_participants"])
    frac_alpha = qc["psd_check"]["fraction_EC_alpha_O_gt_Fp"]
    sig = {e: e09[e]["sigma_mean"] for e in ("1", "3", "8")}
    excl_pct = 100 * qc["n_excluded"] / qc["n_participants"]

    abstract = [
        "Major depressive disorder (MDD) is diagnosed from interviews and rating scales, and resting-state "
        "electroencephalography (EEG) has long been studied as a cheaper, objective complement. Published EEG "
        "classifiers often report accuracies above 95%, but most are trained on small single-site samples and many "
        "split recordings by segment rather than by person, which lets the same individual appear in training and "
        "testing. Hospitals also cannot freely pool EEG and psychiatric labels, which motivates federated learning "
        "(FL) and differential privacy (DP).",
        "This project builds a reproducible, leakage-safe benchmark on the TDBRAIN archive. Two model tracks (log "
        "band power with frontal alpha asymmetry and logistic regression, and EEGNet-8,2 with group normalisation) "
        "are trained under four regimes: centralised, local-only, federated (Flower FedAvg) and federated with "
        "*participant-level* DP, whose privacy unit is the person rather than the 4-second epoch. All splits are made "
        "once, by participant. Privacy is measured with a participant-level likelihood-ratio membership-inference "
        "attack (LiRA), and the frozen models are tested once on the independent Mumtaz data set.",
        f"On {n} adults (MDD vs healthy), the spectral baseline reached a participant-level balanced accuracy of "
        f"{ci(e02)} (AUC {f3(e02['auc'])}); a shuffled-label control gave {f3(e01['balanced_accuracy'])}. Age alone "
        f"reached {f3(e22['balanced_accuracy'])}, so part of the signal is an age confound. FedAvg across four "
        f"simulated clients cost {f3(-c_fl['diff'])} in balanced accuracy and was robust to label skew and client "
        f"dropout, while local-only training lost {f3(e02['balanced_accuracy'] - e05i['balanced_accuracy'])}. "
        f"Participant-level DP at ε = 8 reduced balanced accuracy to {f3(e09['8']['balanced_accuracy'])} and lowered "
        f"the LiRA attack AUC from {f3(L['inf']['auc'])} to {f3(L['8']['auc'])} (chance is 0.5). On Mumtaz the "
        f"models transferred poorly (AUC {f3(x_c['auc'])}–{f3(x_d['auc'])}), and a probe separated the two data sets' "
        f"healthy controls with AUC {f3(probe['auc'])}.",
        "The project is a research benchmark and not a diagnostic tool. The deep-learning track was run only under a "
        "reduced compute budget and its results are preliminary.",
    ]
    keywords = ("EEG, major depressive disorder, federated learning, differential privacy, membership inference, "
                "EEGNet, TDBRAIN, data leakage")

    B = []
    A = B.append
    # =============================================================== CHAPTER 1
    A(("chapter", "Introduction"))
    A(("section", "Overview"))
    A(("p", "Depression is one of the most common causes of disability worldwide [@who]. Its diagnosis rests on "
            "clinical interview and rating scales, which are the foundation of care but take time and are partly "
            "subjective. Resting-state electroencephalography (EEG) is cheap, portable, non-invasive and already "
            "available in many hospitals, and for decades researchers have asked whether measurements taken from it "
            "can support the assessment of depression [@review]."))
    A(("p", "Two practical problems stand between such measurements and any wider use. The first is evidence: many "
            "published EEG classifiers report near-perfect accuracy on small single-site samples, but EEG recordings "
            "are often split into short segments and the segments of one person are distributed between training and "
            "test sets, so the model can recognise the person instead of the condition [@leak]. The second is privacy: "
            "EEG and psychiatric labels are sensitive health data and cannot simply be pooled across institutions "
            "[@kaissis]. Federated learning (FL) trains one model across sites without moving raw data [@fedavg], and "
            "differential privacy (DP) gives a formal bound on what the shared model updates reveal [@dwork]."))
    A(("p", "This project measures what these two protections cost. It builds a reproducible benchmark on the "
            "TDBRAIN archive [@tdbrain] in which every split is made by participant, models are trained centrally, "
            "locally, federated and federated with DP, the privacy gain is measured with an actual attack, and the "
            "frozen models are tested once on an independent data set."))
    A(("section", "Problem Statement"))
    A(("p", "The project addresses the following question: *on TDBRAIN, under subject-level evaluation, how much "
            "accuracy do EEG-based MDD classifiers lose when they are trained federated and with participant-level "
            "differential privacy, how much privacy does that buy against a modern membership-inference attack, and how "
            "well do the models transfer to an independent data set?*"))
    A(("p", "Answering it requires more than a model. The pipeline must (i) build a clean participant-level cohort "
            "from the archive, (ii) freeze the evaluation splits so that no participant appears in two sets or two "
            "clients, (iii) preprocess every recording without statistics shared across people, (iv) train the same "
            "model recipes under four training regimes, (v) account for privacy per participant rather than per "
            "segment, (vi) attack the trained models to measure leakage, and (vii) report every number with a "
            "confidence interval, generated by code from logged runs."))
    A(("section", "EEG and Brain Rhythms"))
    A(("p", "The voltage measured at a scalp electrode is a few tens of microvolts and arises from the summed "
            "activity of large populations of cortical neurons. It is a mixture of rhythms that are conventionally "
            "separated by frequency (⟦fig:fig_rhythms⟧); ⟦tab:bands⟧ lists the bands used in this work."))
    A(("fig", "fig_rhythms", "Illustration of a mixed EEG signal and four of the rhythms into which it is separated. "
            "The traces are drawn mathematically for explanation and are not a recording.", 0.85))
    A(("table", "bands", "Frequency bands used in this work, with commonly described associations",
       ["Band", "Range (Hz)", "Commonly described association (context dependent)"],
       [["Delta (δ)", "1–4", "Prominent in deep sleep"], ["Theta (θ)", "4–8", "Drowsiness and some memory processes"],
        ["Alpha (α)", "8–13", "Relaxed wakefulness, strongest with eyes closed"],
        ["Beta (β)", "13–30", "Alert, active states; also affected by muscle activity"],
        ["Gamma (γ)", "> 30", "Not used: close to the 40 Hz filter edge and contaminated by muscle"]], "llX"))
    A(("p", "These associations depend on age, alertness, medication and recording conditions. They are "
            "background knowledge about EEG and are not interpretations made by the models. In particular, frontal "
            "alpha asymmetry (FAA), often proposed as a marker of depression, did not show a reliable diagnostic effect "
            "in a meta-analysis [@faa]; it is included here as a feature to be tested, not as an assumption."))
    A(("section", "Objectives"))
    A(("p", "The overall objective is a leakage-safe, reproducible benchmark that quantifies the accuracy cost and "
            "the privacy benefit of federated and differentially private training for EEG-based MDD classification. "
            "Three research questions guide the work:"))
    A(("deflist", [("RQ1", "How well can MDD be separated from healthy controls on TDBRAIN when evaluation is strictly "
                           "by participant, and is a compact deep network better than spectral features?"),
                   ("RQ2", "How much accuracy is lost by federated training, by non-identical client data, and by "
                           "participant-level differential privacy at ε ∈ {1, 3, 8}?"),
                   ("RQ3", "How much membership leakage does each regime show under a participant-level LiRA "
                           "attack, and how well do the frozen models transfer to an independent data set?")]))
    A(("p", "The specific objectives that follow from these questions are:"))
    A(("list", True, [
        "Build a participant index from TDBRAIN (first session only, blinded and unknown labels excluded) and define "
        "two tasks: MDD vs healthy (T1) and MDD vs other diagnosed patients (T2).",
        "Freeze participant-level cross-validation splits and simulated client partitions in a hashed file.",
        "Preprocess each recording independently onto the 19 channels shared with the external data set.",
        "Implement a spectral logistic-regression baseline and EEGNet-8,2 with group normalisation.",
        "Train both under centralised, local-only, federated (FedAvg) and federated + participant-level DP regimes.",
        "Audit privacy with a participant-level LiRA membership-inference attack.",
        "Test the frozen models once on Mumtaz and probe whether the two data sets are trivially separable.",
        "Log every run, generate the report and a dashboard from the logs, and package the work for reproduction."]))
    A(("section", "Scope and Significance"))
    A(("p", "The scope is a computational benchmark on existing, de-identified recordings. It does not collect new "
            "data, does not diagnose, and does not recommend treatment. The four clients in the federated experiments "
            "are random partitions of one archive, not real hospitals."))
    A(("p", "The significance of the work lies in its controls. The same frozen splits are used everywhere; a "
            "shuffled-label control checks for leakage; privacy is accounted for the participant and then attacked; "
            "an age-only baseline exposes a demographic confound; and the external test is run once. These controls "
            "matter more than any single accuracy figure, because they decide whether a figure can be trusted."))
    A(("section", "Report Layout"))
    A(("p", "Chapter 2 reviews related work. Chapter 3 presents the proposed system and its design. Chapter 4 "
            "describes the data and methods. Chapter 5 describes the software tools. Chapter 6 reports the results. "
            "Chapter 7 concludes the report and outlines future work."))

    # =============================================================== CHAPTER 2
    A(("chapter", "Related Work"))
    A(("section", "EEG-Based Classification of Depression"))
    A(("p", "Spectral band power, inter-hemispheric asymmetry, connectivity and non-linear measures have all been "
            "studied as features for detecting depression from EEG [@review]. Mumtaz et al. recorded 34 patients with "
            "MDD and 30 healthy controls and reported accuracies close to 98% with band power and alpha asymmetry "
            "[@mumtaz17]; that data set is public [@mumtazdata] and is used in this project only as an external test. "
            "Such results come from small single-centre samples, and the strongest reported effects have not "
            "replicated across data sets. In particular, a meta-analysis of frontal alpha asymmetry found no reliable "
            "diagnostic effect [@faa]."))
    A(("p", "TDBRAIN is a large clinical archive released by Brainclinics: 1,274 participants recorded between 2001 "
            "and 2021, including 426 with MDD and other groups such as ADHD, OCD and subjective memory complaints, with "
            "two minutes each of eyes-open and eyes-closed resting EEG on 26 channels [@tdbrain]. Its size and mixed "
            "clinical population make it a more realistic test bed than single-study samples."))
    A(("section", "Data Leakage in EEG Studies"))
    A(("p", "Brookshire et al. reviewed deep-learning EEG studies and showed that splitting data by segment instead "
            "of by subject inflates accuracy, often dramatically, because a network learns person-specific signatures "
            "[@leak]. Small samples add a second problem: cross-validated estimates from a few hundred participants "
            "carry wide error bars [@varo]. This project therefore splits strictly by participant, reports participant-"
            "level bootstrap confidence intervals, and runs a shuffled-label control."))
    A(("section", "Federated Learning"))
    A(("p", "In federated averaging (FedAvg), a server sends the current model to clients, each client trains on "
            "its own data, and the server averages the returned weights in proportion to client size [@fedavg]. When "
            "client data are not identically distributed, FedAvg can drift; FedProx adds a proximal term that keeps "
            "local models near the global one [@fedprox], and FedBN keeps normalisation layers local [@fedbn]. Federated "
            "learning has been proposed as a route to multi-institution medical models [@rieke], but keeping data on "
            "site does not by itself protect privacy: training data can be reconstructed from shared gradients "
            "[@dlg]."))
    A(("section", "Differential Privacy"))
    A(("p", "A randomised mechanism is (ε, δ)-differentially private if adding or removing one individual changes the "
            "probability of any output by at most a factor e^ε, except with probability δ [@dwork]. DP-SGD makes "
            "training private by clipping each example's gradient and adding Gaussian noise [@abadi]; the total "
            "privacy loss over many steps is tracked with Rényi differential privacy (RDP) [@rdp], converted to "
            "(ε, δ) with a tight bound [@balle]. McMahan et al. showed that the unit of privacy can be the *user* "
            "rather than the example, by clipping and noising each user's whole contribution [@userdp]."))
    A(("p", "The unit matters for EEG. One person contributes many epochs, so a per-epoch guarantee — what a "
            "standard per-sample accountant reports — says little about the person. This project therefore uses "
            "participant-level DP and reports the epoch-level value only as a labelled comparison."))
    A(("section", "Membership Inference"))
    A(("p", "A membership-inference attack asks whether a given record was in a model's training set [@shokri]. The "
            "likelihood-ratio attack (LiRA) of Carlini et al. trains shadow models with and without each target, fits "
            "Gaussians to the target's scores under both hypotheses, and reports the true-positive rate at low "
            "false-positive rates, which is more informative than average accuracy [@lira]. Here the attack is run at "
            "the participant level, so it measures exactly the risk that participant-level DP is meant to bound."))
    A(("section", "Comparison and Research Gap"))
    A(("table", "related", "Position of this project relative to closely related work",
       ["Work", "Data", "Relevance to this project"],
       [["Mumtaz et al. [@mumtaz17]", "64 people, 1 site", "High accuracy from spectral features; used here as the external test"],
        ["Brookshire et al. [@leak]", "Review of EEG DL", "Segment splits inflate accuracy; motivates subject-level splits and E12"],
        ["van der Vinne et al. [@faa]", "Meta-analysis", "FAA is not a reliable marker; tested here as E16"],
        ["McMahan et al. [@fedavg], [@userdp]", "Benchmarks", "FedAvg and user-level DP; adapted to EEG participants"],
        ["Carlini et al. [@lira]", "Benchmarks", "LiRA; adapted to participant-level membership"],
        ["This project", "TDBRAIN + Mumtaz", "Leakage-safe benchmark of FL and participant-level DP with an attack audit and an external test"]],
       "llX"))
    A(("p", "The gap is practical rather than conceptual. Each ingredient exists, but the combination — subject-level "
            "splits, privacy accounted per participant, an attack that tests the guarantee, and an external data set — "
            "has not been applied to EEG-based depression classification on a large clinical archive "
            "(⟦fig:fig_gap⟧)."))
    A(("fig", "fig_gap", "The research gap addressed by this project", 0.95))
    A(("table", "difficulties", "How this project responds to each difficulty",
       ["Difficulty", "Response in this project"],
       [["Segment-level leakage", "Splits by participant, frozen and hashed; shuffled-label control (E01)"],
        ["Optimistic small-sample estimates", "Participant-level bootstrap CIs and paired differences"],
        ["Per-epoch privacy says little about people", "Participant-level DP; epoch-level ε only as comparison"],
        ["A guarantee without evidence", "Participant-level LiRA attack (E10)"],
        ["Results that only hold at one site", "One-time external test on Mumtaz (E11) and a data-set probe (E13)"],
        ["Demographic confounds", "Age-only baseline (E22)"]], "lX"))

    # =============================================================== CHAPTER 3
    A(("chapter", "Proposed System"))
    A(("section", "System Overview"))
    A(("p", "The system is organised in six layers (⟦fig:fig_architecture⟧). The *data layer* builds the participant "
            "index and the frozen splits. The *preprocessing layer* turns each recording into clean 4-second epochs on "
            "19 channels. The *model layer* holds the two tracks. The *regime layer* trains them centrally, locally, "
            "federated or federated with DP. The *evaluation layer* scores them per participant, attacks them and runs "
            "the external test. Tracking and reporting connect everything through MLflow."))
    A(("fig", "fig_architecture", "Layered architecture of the proposed system", 0.95))
    A(("section", "Design Principles"))
    A(("list", False, [
        "**One split, made once.** Participants are split once and the file is hashed; every experiment reads it.",
        "**Nothing is fitted on test data.** Scalers, thresholds, hyperparameters and early stopping use training or "
        "inner-validation participants only.",
        "**The privacy unit is the participant.** Sampling, clipping and accounting all operate on people.",
        "**The external set is touched once.** Only one module may read Mumtaz, and it logs every access.",
        "**Every number comes from a logged run.** Reports and the dashboard are generated, never typed.",
        "**Data never leave the local folder.** Raw EEG is never committed, uploaded or shown."]))
    A(("section", "System Requirements"))
    A(("table", "fr", "Functional requirements",
       ["ID", "The system shall", "How it is met"],
       [["FR1", "build a first-session participant index with task labels", "src/data/index.py"],
        ["FR2", "freeze participant-level splits and client partitions", "splits.json with SHA-256 hash"],
        ["FR3", "preprocess each recording without cross-subject statistics", "src/preprocess/pipeline.py"],
        ["FR4", "train two model tracks under four regimes", "src/train, src/fl"],
        ["FR5", "account privacy per participant", "src/privacy/participant_dp.py, accountant.py"],
        ["FR6", "audit membership leakage", "src/privacy/lira.py"],
        ["FR7", "test frozen models once on an external set", "src/eval/external.py"],
        ["FR8", "log runs and generate the report and dashboard", "MLflow, src/report, dashboard/"]], "llX"))
    A(("table", "nfr", "Non-functional requirements",
       ["Quality", "Requirement"],
       [["Data governance", "Raw EEG stays local; data folders are git-ignored; only aggregate results are reported"],
        ["Leakage control", "No participant in two sets or two clients; checked by automated tests"],
        ["Reproducibility", "Fixed seeds; config, split, data-table and git hashes stored with every run"],
        ["Hardware", "Runs on a CPU; uses CUDA or Apple MPS only if present"],
        ["Honesty", "Synthetic, budget-reduced and preliminary results are labelled wherever they appear"],
        ["Safety of wording", "No text claims diagnostic use or calls a feature a biomarker of depression"]], "lX"))
    A(("section", "Processing Pipeline"))
    A(("p", "⟦fig:fig_pipeline⟧ follows one recording through preprocessing. Each step operates on that recording "
            "alone, so no information passes between participants before the split is applied. A recording with too "
            "few clean epochs is excluded with its reason recorded; it is never silently dropped."))
    A(("fig", "fig_pipeline", "Flowchart of the preprocessing pipeline for one recording", 0.62))
    A(("section", "Training Regimes"))
    A(("p", "Each model recipe is trained in four ways (⟦fig:fig_regimes⟧). In the *centralised* regime all "
            "training participants are pooled, which gives an upper reference. In the *local-only* regime each "
            "simulated client trains alone, which represents a single hospital. In the *federated* regime the clients "
            "train locally and a server averages their weights. In the *federated + DP* regime each client clips the "
            "contribution of every participant and adds Gaussian noise before its update leaves the client."))
    A(("fig", "fig_regimes", "The four training regimes. C1–C4 are the simulated clients", 0.98))
    A(("section", "Implementation Status"))
    A(("table", "status", "Status of the main components at the time of this report",
       ["Component", "Status"],
       [["Participant index, splits, preprocessing", "Implemented and run on the full TDBRAIN release"],
        ["Spectral LR, all four regimes", "Implemented; all essential experiments run (5 folds)"],
        ["EEGNet-8,2", "Implemented; run on fold 0 only under a reduced compute budget (preliminary)"],
        ["Participant-level DP and accounting", "Implemented, unit-tested and run for LR at ε ∈ {1, 3, 8}"],
        ["LiRA audit", "Run for LR, centralised and federated, at every ε"],
        ["External test and data-set probe", "Run once for LR models"],
        ["Report, dashboard, tests, packaging", "Implemented; 42 automated tests pass"],
        ["Segment-split demonstration (E12), FedProx/FedBN, optional ablations", "Implemented; not yet run on real data"]],
       "lX"))

    # =============================================================== CHAPTER 4
    A(("chapter", "Material and Methodologies"))
    A(("section", "Material"))
    A(("p", f"Two data sets were used ({tab_ref('data')}). **TDBRAIN** (release V3) was obtained under its data-use "
            f"agreement and contains BioSemi BDF recordings in a BIDS layout with 26 EEG channels plus "
            f"eye, muscle and status channels at 500 Hz, and a participant table with {c['rows']} session rows. "
            f"**Mumtaz** is a public data set of 34 patients with MDD and 30 healthy controls recorded on 19 channels at "
            f"256 Hz with a linked-ear reference [@mumtazdata]."))
    A(("table", "data", "Data used in this project",
       ["Property", "TDBRAIN (training and cross-validation)", "Mumtaz (external test only)"],
       [["Participants", f"{c['first_session']} (first session)", "34 MDD, 30 healthy"],
        ["Channels", "26 EEG (+ 7 non-EEG)", "19 EEG (+ 3 non-EEG)"],
        ["Sampling rate", "500 Hz", "256 Hz"],
        ["Recording used", "2 min eyes closed (restEC)", "Eyes closed (EC)"],
        ["Labels", "Clinical indication; replication sample blinded", "MDD or healthy, from file names"],
        ["Access", "ORCID login and data-use agreement", "Public (figshare)"]], "lXX"))
    A(("p", "The pipeline uses the 19 channels that both data sets share (⟦fig:fig_montage⟧). In TDBRAIN's naming, "
            "Mumtaz's T3, T4, T5 and T6 are T7, T8, P7 and P8. Missing channels are never interpolated: using the "
            "intersection avoids inventing signal."))
    A(("fig", "fig_montage", "The 19 channels used (filled) and the TDBRAIN-only channels not used (grey), top view, "
            "nose upwards", 0.5))
    A(("section", "Cohort and Task Definitions"))
    A(("p", f"Only each participant's first session was kept, because later sessions may follow treatment. "
            f"Participants whose indication was missing or UNKNOWN ({c['unknown']}) were excluded — UNKNOWN does not mean "
            f"healthy — and the blinded replication sample is distributed separately. Adults aged 18–65 were kept. Two "
            f"tasks were defined from the primary indication (⟦fig:fig_cohort⟧): **T1**, MDD vs HEALTHY "
            f"(n = {c['T1_n']}), the primary task, and **T2**, MDD vs other diagnosed patients (n = {c['T2_n']})."))
    A(("fig", "fig_cohort", "Cohort flow from the participant table to the two tasks", 0.7))
    A(("table", "cohort", "Cohort per task and class (sex given as the raw code of the participant table)",
       ["Task", "Class", "n", "Age, mean (SD)", "Sex code 0 / 1", "Comorbid"],
       [["T1", "MDD", str(t1m["n"]), f"{t1m['age_mean']:.1f} ({t1m['age_sd']:.1f})",
         f"{t1m['sex_raw_counts'].get('0', 0)} / {t1m['sex_raw_counts'].get('1', 0)}", str(t1m["comorbid"])],
        ["T1", "Healthy", str(t1h["n"]), f"{t1h['age_mean']:.1f} ({t1h['age_sd']:.1f})",
         f"{t1h['sex_raw_counts'].get('0', 0)} / {t1h['sex_raw_counts'].get('1', 0)}", str(t1h["comorbid"])],
        ["T2", "Other patients", str(t2o["n"]), f"{t2o['age_mean']:.1f} ({t2o['age_sd']:.1f})",
         f"{t2o['sex_raw_counts'].get('0', 0)} / {t2o['sex_raw_counts'].get('1', 0)}", str(t2o["comorbid"])]],
       "llrrrr"))
    A(("p", f"Healthy participants were on average {t1m['age_mean'] - t1h['age_mean']:.1f} years younger than "
            f"those with MDD. Because EEG power changes with age, an age-only baseline was added as a control (E22). "
            f"The 'other patients' group of T2 is broad: the largest subgroups were ADHD ({t2o['groups'].get('ADHD', 0)}), "
            f"SMC ({t2o['groups'].get('SMC', 0)}) and OCD ({t2o['groups'].get('OCD', 0)})."))
    A(("section", "Participant-Level Splits"))
    A(("p", "Splits were made once with stratified group 5-fold cross-validation, grouped by participant and "
            "stratified by label and age tertile (⟦fig:fig_splits⟧). Inside each outer training fold, 20% of the "
            "participants form an inner validation set used for hyperparameters, decision thresholds and early "
            "stopping. The remaining training participants are assigned to K = 4 clients, either at random (IID) or "
            "with label skew drawn from a Dirichlet distribution (α = 0.5) with at least five participants per class "
            "per client. The whole structure is saved to one file with a SHA-256 hash."))
    A(("fig", "fig_splits", "Participant-level splitting scheme", 0.95))
    A(("section", "Preprocessing"))
    A(("table", "prep", "Preprocessing steps and settings",
       ["Step", "Setting"],
       [["Channels", "Non-EEG channels dropped; T3/T4/T5/T6 renamed; 19 channels picked in a fixed order"],
        ["Units", "Converted from volts to microvolts"],
        ["Filtering", "Zero-phase FIR band-pass 1–40 Hz; 50 Hz notch"],
        ["Bad channels", "Flat or outlying channels rebuilt by spherical-spline interpolation within the recording "
                         "(at most 3; count logged)"],
        ["Reference", "Common average over the 19 channels"],
        ["Resampling", "256 Hz"],
        ["Epochs", "4 s (1,024 samples), non-overlapping"],
        ["Rejection", "Peak-to-peak above 150 µV or below 1 µV on any channel"],
        ["Inclusion", "At least 10 clean epochs; the first 30 clean epochs kept"]], "lX"))
    A(("p", f"Of {qc['n_participants']} recordings with a task label, {qc['n_excluded']} ({excl_pct:.1f}%) were "
            f"excluded for too few clean epochs, {100 * qc['epoch_rejection_rate']:.1f}% of epochs were rejected, and "
            f"{qc['n_channels_interpolated_total']} channels were interpolated in {qc['n_recordings_with_interpolation']} "
            f"recordings. As a sanity check, eyes-closed alpha power at O1/O2 exceeded that at Fp1/Fp2 in "
            f"{100 * frac_alpha:.0f}% of participants, as expected from the posterior alpha rhythm (⟦fig:fig_psd⟧)."))
    A(("fig", "fig_psd", "Group-mean eyes-closed power spectrum (19-channel mean, shaded 95% interval of the mean) on "
            "TDBRAIN. Dotted lines mark the band edges", 0.8))
    A(("section", "Spectral Features (Track A)"))
    A(("p", "For each epoch, Welch's method [@welch] with 2-second windows and 50% overlap estimates the power "
            "spectrum of each channel. Absolute power is integrated over the delta, theta, alpha and beta bands and "
            "log-transformed, giving 19 × 4 = 76 features; frontal alpha asymmetry adds one more:"))
    A(("eq", r"\mathrm{FAA} = \ln P_\alpha(\mathrm{F4}) - \ln P_\alpha(\mathrm{F3})",
       "FAA = ln Pα(F4) − ln Pα(F3)"))
    A(("p", "Features are averaged over each participant's epochs. The centralised model is a standard scaler "
            "fitted on training participants, followed by L2-regularised logistic regression with balanced class "
            "weights [@sklearn]; the regularisation strength C ∈ {0.01, 0.1, 1, 10} is chosen by inner-validation AUC "
            "and the decision threshold is then tuned on the same validation participants for balanced accuracy."))
    A(("section", "EEGNet (Track B)"))
    A(("p", "Track B is EEGNet-8,2 [@eegnet] with group normalisation [@gn] in place of batch normalisation, because "
            "batch statistics mix participants and are awkward under federation and differential privacy "
            "(⟦fig:fig_eegnet⟧). The network has 2,946 trainable parameters. Each recording is z-scored per channel "
            "using its own statistics only. Training uses Adam [@adam] (learning rate 10⁻³, batch 64) and "
            "class-weighted cross-entropy; early stopping monitors participant-level balanced accuracy on the inner "
            "validation set. A participant's probability is the mean of their epoch probabilities."))
    A(("fig", "fig_eegnet", "EEGNet-8,2 with group normalisation as used in this project", 0.98))
    A(("section", "Federated Learning"))
    A(("p", "Federated training uses the Flower simulation engine [@flower] with one client per partition. Each round "
            "the server sends the global weights, every available client trains locally, and the server forms the "
            "weighted FedAvg [@fedavg]:"))
    A(("eq", r"w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j}\, w_{t+1}^{(k)}",
       "w(t+1) = Σ over available clients k of [n_k / Σ n_j] · w_k(t+1)"))
    A(("p", "where n_k is the size of client k and S_t the clients available in round t. After every round the "
            "server evaluates the global model on the inner-validation participants and keeps the best round. For "
            "the linear model the feature scaler is computed from client sufficient statistics (sums and sums of "
            "squares), which equals the pooled training scaler without sharing raw features. A seeded dropout test "
            "makes each client unavailable with probability 0.2 per round. An equivalent in-process engine with the "
            "same client, selection and aggregation code was verified to produce identical weights and is used for "
            "speed."))
    A(("section", "Participant-Level Differential Privacy"))
    A(("p", "Standard per-sample DP would treat each 4-second epoch as an individual. Here the participant is the "
            "unit (⟦fig:fig_fl_dp⟧). At each local step a client samples each of its participants independently with "
            "probability q = 0.25, computes the gradient of each sampled participant's loss on one randomly chosen "
            "epoch, clips it to norm C = 1, and releases"))
    A(("eq", r"\tilde g = \frac{1}{qN}\Big(\sum_{i \in B} g_i \min\big(1, \tfrac{C}{\lVert g_i \rVert_2}\big) + "
             r"\mathcal{N}(0, \sigma^2 C^2 I)\Big)",
       "g̃ = (1 / qN) · ( Σ over sampled i of clip(g_i, C) + N(0, σ²C²I) )"))
    A(("p", f"where N is the number of the client's participants. Per-participant gradients are obtained with "
            f"vectorised automatic differentiation; the RDP accountant of Opacus [@opacus], [@rdp] composes the steps "
            f"of every round, and the reported ε is the maximum over clients at δ = 10⁻⁵. For each target ε the noise "
            f"multiplier σ is found by bisection; for logistic regression this gave σ ≈ {sig['8']:.2f} (ε = 8), "
            f"{sig['3']:.2f} (ε = 3) and {sig['1']:.2f} (ε = 1). A pooled feature scaler would leak information "
            f"outside the accounting, so DP logistic regression uses a per-participant transform instead (log band "
            f"power relative to the participant's own channel mean, plus relative band levels), applied at every ε "
            f"including the non-private reference so that the comparison is like for like."))
    A(("fig", "fig_fl_dp", "One round of federated training with participant-level differential privacy", 0.95))
    A(("section", "Privacy Audit (LiRA)"))
    A(("p", "The threat model is an honest-but-curious server, or anyone able to query the released model, who wants "
            "to know whether a given person's EEG was used in training. The audit follows LiRA [@lira] at the "
            "participant level (⟦fig:fig_lira⟧). S models are trained with the target recipe, each on a random half of "
            "the labelled T1 pool; each participant's score is the mean, over their epochs, of the logit of the "
            "correct-class probability. Each model in turn is the target: for every participant the scores of the "
            "other models are split into members and non-members, Gaussians with a shared variance are fitted, and the "
            "likelihood ratio of the target's score is the attack statistic. The ROC, its AUC and the true-positive "
            "rates at 1% and 5% false-positive rate are reported; the rate at 0.1% is reported only with at least "
            "1,000 non-member evaluations."))
    A(("fig", "fig_lira", "Participant-level likelihood-ratio membership-inference audit", 0.95))
    A(("section", "Evaluation and Statistics"))
    A(("p", "All metrics are computed per participant. Balanced accuracy and AUC are primary; sensitivity, "
            "specificity, F1, Matthews correlation and expected calibration error are also logged. Out-of-fold "
            "predictions are averaged over seeds per participant, and 95% confidence intervals come from 2,000 "
            "class-stratified bootstrap resamples of participants [@boot]. Comparisons between configurations use a "
            "paired bootstrap of the difference on the same participants. No p-values from five-fold tests are "
            "reported as significant."))
    A(("section", "External Test and Data-Set Probe"))
    A(("p", "For the external test, each model is retrained on all T1 participants with the hyperparameters chosen "
            "in cross-validation, and its decision threshold is fixed from TDBRAIN validation. Only then are the "
            "Mumtaz eyes-closed recordings loaded, preprocessed with the identical configuration, and scored once. A "
            "marker file prevents a second evaluation. The dataset-identity probe (E13) trains a logistic regression "
            "on band-power features to tell TDBRAIN healthy controls from Mumtaz healthy controls under grouped "
            "cross-validation; a high AUC means the two data sets differ in ways unrelated to diagnosis."))
    A(("table", "experiments", "Experiment matrix (E22 was added during the project)",
       ["ID", "Question", "Model", "Regime", "Status"],
       [["E01", "Shuffled-label control (≈ 0.5)", "LR, EEGNet", "Centralised", "Run (EEGNet fold 0)"],
        ["E02", "Spectral baseline (T1)", "LR", "Centralised", "Run"],
        ["E03", "Harder contrast (T2)", "LR", "Centralised", "Run"],
        ["E04", "Deep vs features", "EEGNet", "Centralised", "Run (fold 0)"],
        ["E05", "Single-site loss", "LR", "Local only", "Run"],
        ["E06", "FL cost, IID vs label skew, dropout", "LR", "FedAvg", "Run"],
        ["E09", "Privacy cost, ε ∈ {1, 3, 8, ∞}", "LR", "FedAvg + DP", "Run"],
        ["E10", "LiRA audit", "LR", "Central, federated", "Run"],
        ["E11", "External transfer", "LR", "Frozen", "Run once"],
        ["E13", "Dataset-identity probe", "LR", "Centralised", "Run"],
        ["E16", "FAA alone", "LR", "Centralised", "Run"],
        ["E22", "Age alone (confound control)", "LR", "Centralised", "Run"],
        ["E07, E08, E12, E14–E21", "FedProx, FedBN, leakage demo, ablations", "Both", "Various", "Implemented, not run"]],
       "lXllX"))

    # =============================================================== CHAPTER 5
    A(("chapter", "Programming Language and Tools Used"))
    A(("section", "Programming Language and Libraries"))
    pk = V.get("packages") or {}
    A(("p", "The system is written in Python 3.11 with a locked environment managed by uv. ⟦tab:tools⟧ lists the main "
            "libraries and their roles."))
    A(("table", "tools", "Software tools and their roles",
       ["Tool", "Version", "Role"],
       [["MNE-Python [@mne]", pk.get("mne", ""), "Reading BDF/EDF, filtering, referencing, interpolation"],
        ["NumPy, SciPy, pandas", f"{pk.get('numpy', '')}, {pk.get('scipy', '')}, {pk.get('pandas', '')}", "Arrays, Welch PSD, tables"],
        ["scikit-learn [@sklearn]", pk.get("scikit-learn", ""), "Splits, logistic regression, metrics"],
        ["PyTorch [@torch]", pk.get("torch", ""), "EEGNet, linear model, per-participant gradients"],
        ["Flower [@flower]", pk.get("flwr", ""), "Federated simulation (ClientApp / ServerApp)"],
        ["Opacus [@opacus]", pk.get("opacus", ""), "RDP privacy accountant"],
        ["MLflow [@mlflow]", pk.get("mlflow", ""), "Run tracking, artefacts"],
        ["Streamlit, Plotly", f"{pk.get('streamlit', '')}, {pk.get('plotly', '')}", "Read-only results dashboard"],
        ["pytest, ruff", "—", "Automated tests and linting"]], "llX"))
    A(("section", "Software Design"))
    A(("p", "The code follows the layers of ⟦fig:fig_architecture⟧. Configuration lives in YAML files (data, "
            "preprocessing, models, federated learning, DP, audit) and in one file per experiment; an experiment "
            "runner expands each configuration into runs over variants, models, ε values, folds and seeds. Named "
            "overlays (smoke, budget, fast, demo15) reduce the compute budget for quick passes and are recorded as a "
            "tag on every run."))
    A(("table", "modules", "Main modules",
       ["Module", "Responsibility"],
       [["src/data", "Participant index, frozen splits, TDBRAIN and Mumtaz readers, downloads, synthetic test data"],
        ["src/preprocess", "Per-recording pipeline and QC report"],
        ["src/features, src/models", "Welch band power and FAA; logistic regression and EEGNet"],
        ["src/train, src/fl", "Centralised and local training; Flower client, server, strategy"],
        ["src/privacy", "Participant-level DP-SGD, RDP accountant, LiRA"],
        ["src/eval", "Metrics, bootstrap, external test (the only Mumtaz reader)"],
        ["src/explain, src/report", "Coefficients, spatial patterns, band occlusion; export, tables, figures"],
        ["dashboard/", "Eight-page Streamlit application"]], "lX"))
    A(("section", "Data Governance and Privacy Measures"))
    A(("list", False, [
        "TDBRAIN was obtained by the author under its data-use agreement; the copy used was the author's own private "
        "upload, checked to be private before download, and only the needed files were extracted.",
        "Data folders, caches and EEG file types are excluded from version control, and the Docker image contains no data.",
        "Credentials are stored outside the code and never printed or logged.",
        "Only aggregate results appear in this report and the repository; per-participant predictions stay local.",
        "A static test checks that no training module reads the external data set."]))
    A(("section", "Experiment Tracking and Reproducibility"))
    A(("p", f"Every run is logged to MLflow with its configuration, configuration hash, split hash, participant-table "
            f"hash, git commit, seed, metrics, confusion matrix, ε, checkpoint and per-participant predictions. An "
            f"export step aggregates the runs into result tables, from which the report and the dashboard are "
            f"generated. The results in Chapter 6 come from {V['n_runs']} logged runs."))
    A(("section", "The Dashboard"))
    A(("p", "A read-only Streamlit dashboard presents the results in eight pages: overview, cohort and QC, model "
            "performance, federated learning, privacy, external test, explainability and reproducibility. It needs no "
            "credentials, accepts no uploads, and reads only the exported result files (⟦fig:dash_privacy⟧)."))
    A(("fig", "dash_privacy", "The privacy page of the dashboard, generated from the logged TDBRAIN runs", 0.8))
    A(("section", "Testing"))
    A(("table", "tests", "Automated tests (42 in total, all passing)",
       ["Area", "What is checked"],
       [["Splits", "No participant in two sets or two clients; one test fold each; stable hash; tamper detection"],
        ["Preprocessing", "Shape (n, 19, 1024), channel order, microvolt scale, 256 Hz, no NaN, renaming, BDF files"],
        ["No test fitting", "Scaler, threshold and early stopping receive only training or validation participants"],
        ["Differential privacy", "Clipping bound per participant, noise scale, accountant against an analytic value"],
        ["Federated learning", "K = 1 federated equals centralised; Flower engine equals the in-process engine"],
        ["LiRA", "AUC > 0.9 when members are separable; ≈ 0.5 with no membership effect"],
        ["Isolation and controls", "Only the external module reads Mumtaz; the E01 check fails on a leaky positive control"]],
       "lX"))

    # =============================================================== CHAPTER 6
    A(("chapter", "Results and Analysis"))
    A(("section", "Experimental Setup"))
    A(("p", "All experiments ran on a 4-core CPU without a graphics card. Logistic-regression experiments used the "
            "full five folds; E01–E05, E16 and E22 used three seeds and E06, E09 and E10 one seed with 20 federated "
            "rounds and 16 LiRA models per configuration. EEGNet was run only on fold 0 with at most 15 passes (E01) "
            "or 8 passes (E04), because one model took several minutes on this hardware; its results are therefore "
            "preliminary. All figures and tables below are generated from the logged runs."))
    A(("section", "Leakage Control"))
    A(("p", f"With shuffled training labels the spectral model reached a balanced accuracy of {ci(e01)} and an AUC "
            f"of {f3(e01['auc'])} on {int(e01['n_participants'])} participants, and EEGNet "
            f"{f3(e01n['balanced_accuracy'])} on fold 0. Both are consistent with chance, so the pipeline does not leak "
            f"label information into the test folds."))
    A(("section", "Centralised Results"))
    rows_main = []
    for lab_, r in [("E01 shuffled labels", e01), ("E02 band power + FAA", e02), ("E16 FAA only", e16),
                    ("E22 age only", e22), ("E03 T2: MDD vs other patients", e03), ("E04 EEGNet (fold 0)", e04)]:
        rows_main.append([lab_, str(int(r["n_participants"])), ci(r), ci(r, "auc"), f3(r["sensitivity"]),
                          f3(r["specificity"])])
    A(("table", "central", "Centralised results (participant level, 95% bootstrap CI)",
       ["Configuration", "n", "Balanced accuracy", "AUC", "Sens.", "Spec."], rows_main, "XrllXX"))
    A(("p", f"The spectral baseline reached {ci(e02)} balanced accuracy on T1 (⟦fig:fig_main_results⟧). Two "
            f"controls qualify this. First, age alone reached {f3(e22['balanced_accuracy'])}; the band-power model was "
            f"better by {f3(c_age['diff'])} [{f3(c_age['lo'])}, {f3(c_age['hi'])}], so it carries information beyond age, "
            f"but part of its performance is likely age-related. Second, FAA alone was close to chance "
            f"({f3(e16['balanced_accuracy'])}), {f3(-c_faa['diff'])} below the full model, which agrees with the "
            f"meta-analysis [@faa]. The harder T2 contrast fell to {f3(e03['balanced_accuracy'])}, showing that the "
            f"features separate depression from health much better than from other clinical groups."))
    A(("fig", "fig_main_results", "Participant-level balanced accuracy and AUC with 95% bootstrap intervals", 0.98))
    A(("p", f"On fold 0, EEGNet reached {f3(e04['balanced_accuracy'])} balanced accuracy and AUC "
            f"{f3(e04['auc'])}. Paired with the spectral model on the same {int(c_eeg['n_participants'])} participants, "
            f"the difference was {f3(c_eeg['diff'])} [{f3(c_eeg['lo'])}, {f3(c_eeg['hi'])}]: under this budget the deep "
            f"model did not outperform the features, and the interval is too wide for a conclusion."))
    A(("fig", "fig_roc", "ROC curves of the logistic-regression configurations (seed-averaged out-of-fold "
            "predictions)", 0.55))
    A(("section", "Federated and Local Training"))
    A(("table", "fl", "Federated and local-only logistic regression on T1",
       ["Regime", "Balanced accuracy", "AUC", "Worst client BA"],
       [["Centralised", ci(e02), ci(e02, "auc"), "—"],
        ["FedAvg, IID", ci(e06i), ci(e06i, "auc"), "—"],
        ["FedAvg, label skew", ci(e06s), ci(e06s, "auc"), "—"],
        ["FedAvg, IID, 20% client dropout", ci(e06d), ci(e06d, "auc"), "—"],
        ["Local only, IID (mean client)", ci(e05i), ci(e05i, "auc"), f3(e05i.get("worst_balanced_accuracy"))],
        ["Local only, label skew (mean client)", ci(e05s), ci(e05s, "auc"), f3(e05s.get("worst_balanced_accuracy"))]],
       "Xllr"))
    A(("p", f"FedAvg lost {f3(-c_fl['diff'])} [{f3(-c_fl['hi'])}, {f3(-c_fl['lo'])}] in balanced accuracy against "
            f"centralised training, with no measurable loss in AUC ({f3(c_fla['diff'])}). Label skew changed the "
            f"federated result by only {f3(c_sk['diff'])}, and random client dropout had no visible effect. Training "
            f"on a single simulated site, in contrast, lowered balanced accuracy to {f3(e05i['balanced_accuracy'])} "
            f"(IID) and {f3(e05s['balanced_accuracy'])} (label skew) (⟦fig:fig_fl⟧). For a linear model on these "
            f"features, federation therefore recovers most of what pooling would give, which is the practical case for "
            f"it."))
    A(("fig", "fig_fl", "Centralised, federated and local-only logistic regression (worst client marked for "
            "local-only)", 0.9))
    A(("section", "Cost of Participant-Level Privacy"))
    A(("table", "dp", "Federated logistic regression with participant-level DP (δ = 10⁻⁵)",
       ["Target ε", "Achieved ε", "σ", "Balanced accuracy", "AUC", "Δ BA vs ε = ∞"],
       [[e, f2(e09[e].get("epsilon_mean")), f2(e09[e].get("sigma_mean")), ci(e09[e]), ci(e09[e], "auc"),
         f"{f3(c_dp[e]['diff'])} [{f3(c_dp[e]['lo'])}, {f3(c_dp[e]['hi'])}]"] for e in ("8", "3", "1")]
       + [["∞", "—", "—", ci(e09["inf"]), ci(e09["inf"], "auc"), "—"]], "rrrXXX"))
    A(("p", f"Participant-level DP cost between {f3(-c_dp['8']['diff'])} (ε = 8) and {f3(-c_dp['3']['diff'])} "
            f"(ε = 3) in balanced accuracy relative to the non-private reference with the same features "
            f"(⟦fig:fig_privacy⟧). Accuracy remained clearly above chance even at ε = 1. Because each participant "
            f"contributes a single averaged feature vector to this model, an epoch-level ε is not defined for it; for "
            f"EEGNet it would be much smaller than the participant-level value and would overstate the protection."))
    A(("fig", "fig_privacy", "Privacy–utility trade-off: accuracy (left) and LiRA attack AUC (right) against "
            "participant-level ε", 0.98))
    A(("section", "Privacy Audit"))
    A(("table", "lira", "Participant-level LiRA audit of logistic regression (0.5 = chance)",
       ["Regime", "ε", "Attack AUC", "TPR @ 1% FPR", "TPR @ 5% FPR", "Sanity AUC"],
       [["Centralised", "∞", f3(Lc["auc"]), f3(Lc["tpr_at_fpr_0.01"]), f3(Lc["tpr_at_fpr_0.05"]),
         f3(Lc.get("sanity_auc_nonmembers_only"))]]
       + [["Federated", "∞" if e == "inf" else e, f3(L[e]["auc"]), f3(L[e]["tpr_at_fpr_0.01"]),
           f3(L[e]["tpr_at_fpr_0.05"]), f3(L[e].get("sanity_auc_nonmembers_only"))] for e in ("inf", "8", "3", "1")],
       "llrrrr"))
    A(("p", f"Without DP the attack had a modest but real advantage: AUC {f3(Lc['auc'])} for the centralised model "
            f"and {f3(L['inf']['auc'])} for the federated one. With participant-level DP the advantage almost "
            f"disappeared (AUC {f3(L['8']['auc'])} at ε = 8 and {f3(L['3']['auc'])}–{f3(L['1']['auc'])} at "
            f"ε ≤ 3), and the true-positive rate at 1% FPR fell to the level expected by chance "
            f"(⟦fig:fig_lira_roc⟧). The sanity check — relabelling a random half of non-members as members — gave "
            f"AUC close to 0.5 everywhere. With {int(L['inf']['n_nonmember_evals'])} non-member evaluations per "
            f"configuration, the rate at 0.1% FPR could also be estimated. Federation alone reduced leakage only "
            f"slightly; the formal guarantee came from DP."))
    A(("fig", "fig_lira_roc", "LiRA ROC on log–log axes for logistic regression", 0.55))
    A(("section", "External Test and Data-Set Probe"))
    A(("table", "external", f"One-time external test on Mumtaz eyes-closed recordings (n = {int(x_c['n'])})",
       ["Model", "Balanced accuracy", "AUC", "Sensitivity", "Specificity"],
       [[nm, ci(r), ci(r, "auc"), f3(r["sensitivity"]), f3(r["specificity"])]
        for nm, r in (("LR centralised", x_c), ("LR FedAvg", x_f), ("LR FedAvg + DP, ε = 8", x_d))], "lllrr"))
    A(("p", f"Transfer to Mumtaz was weak (⟦fig:fig_external⟧). The AUCs ({f3(x_c['auc'])}–{f3(x_d['auc'])}) were "
            f"close to chance with wide intervals, and the fixed TDBRAIN thresholds labelled almost every Mumtaz "
            f"participant as MDD (sensitivity {f3(x_c['sensitivity'])}, specificity {f3(x_c['specificity'])}), so the "
            f"balanced accuracies should not be read as successful transfer. The data-set probe explains why: band "
            f"power separated TDBRAIN and Mumtaz healthy controls with AUC {ci(probe, 'auc')}. Amplifier, reference, "
            f"country and population differ so much that a model trained on one archive meets a shifted feature "
            f"distribution on the other. This is the main reason the design keeps Mumtaz strictly external rather than "
            f"pooling it."))
    A(("fig", "fig_external", "TDBRAIN cross-validated AUC against Mumtaz external AUC for the same model recipes", 0.85))
    A(("section", "Explainability"))
    A(("p", "⟦fig:fig_explain⟧ shows the standardised coefficients of the spectral model, averaged over folds and "
            "grouped by band and region, for features whose sign was stable in at least four of five folds. These "
            "describe what the model uses on this data set and explain the model, not the disorder: correlated "
            "features share weight, and age may be reflected in them."))
    A(("fig", "fig_explain", "Stable logistic-regression coefficients by band and region (explains the model, not "
            "the disorder)", 0.7))
    A(("section", "Discussion of the Research Questions"))
    A(("table", "rq", "What the present results say about each research question",
       ["", "Finding", "Present answer"],
       [["RQ1", f"Spectral LR reached {f3(e02['balanced_accuracy'])} BA on T1 under participant-level evaluation; part "
                f"of this is age. EEGNet did not outperform it on fold 0.",
         "Moderate separation; deep vs features open"],
        ["RQ2", f"FedAvg cost {f3(-c_fl['diff'])} BA and was robust to skew and dropout; DP cost "
                f"{f3(-c_dp['8']['diff'])}–{f3(-c_dp['3']['diff'])} BA at ε = 8 to 3.",
         "Federation cheap; DP costs about 0.1"],
        ["RQ3", f"LiRA AUC fell from {f3(L['inf']['auc'])} to ≈ 0.5 with DP. External AUC was near chance and the data "
                f"sets were separable (probe AUC {f3(probe['auc'])}).",
         "DP removes measurable leakage; transfer fails"]], "lXl"))

    # =============================================================== CHAPTER 7
    A(("chapter", "Conclusion and Future Work"))
    A(("section", "Conclusion"))
    A(("p", "This project designed and built a leakage-safe benchmark for EEG-based MDD classification on a large "
            "clinical archive, and used it to measure the cost and benefit of federated and differentially private "
            "training. The main outcomes are:"))
    A(("list", True, [
        f"A reproducible pipeline from the TDBRAIN release to frozen participant-level splits and clean epochs, which "
        f"processed {qc['n_participants'] - qc['n_excluded']} of {qc['n_participants']} labelled recordings and passed a "
        f"shuffled-label leakage control.",
        f"A spectral baseline with balanced accuracy {f3(e02['balanced_accuracy'])} on MDD vs healthy, shown by an "
        f"age-only control to be partly but not wholly explained by age, and a much weaker separation of MDD from "
        f"other patients ({f3(e03['balanced_accuracy'])}).",
        f"Federated training that lost only {f3(-c_fl['diff'])} in balanced accuracy and was robust to label skew and "
        f"client dropout, while single-site training lost {f3(e02['balanced_accuracy'] - e05i['balanced_accuracy'])}.",
        "Participant-level differential privacy that cost about 0.09–0.11 in balanced accuracy and reduced a "
        "participant-level membership-inference attack to chance.",
        f"A one-time external test showing poor transfer to Mumtaz, explained by a near-perfect separation of the two "
        f"data sets (probe AUC {f3(probe['auc'])})."]))
    A(("p", "The project therefore delivers a working, tested benchmark and a clear answer for the linear model: "
            "federation is cheap, participant-level privacy costs a measurable amount of accuracy and removes "
            "measurable leakage, and cross-data-set transfer remains the hardest problem. It is not a diagnostic tool."))
    A(("section", "Limitations"))
    A(("list", False, [
        "The simulated clients are random partitions of one archive, not hospitals with different hardware and populations.",
        "EEGNet was run only on one fold under a reduced budget; E12, FedProx, FedBN and the ablations were not run.",
        "Federated, DP and audit experiments used one seed; the external set is small (58 participants with eyes-closed data).",
        "Healthy participants were younger than those with MDD, and age is partly confounded with the label.",
        "Medication was not systematically recorded in TDBRAIN, and clinical labels carry noise and comorbidity.",
        "The sex coding of the participant table was not confirmed, so sex is reported only as raw codes.",
        "This is a research benchmark and not a diagnostic tool."]))
    A(("section", "Future Work"))
    A(("list", True, [
        "**Complete the deep-learning track.** Run EEGNet under the full specification (5 folds, 3 seeds, up to 100 "
        "epochs), with FedProx, FedBN, DP and LiRA, on a machine with a graphics card.",
        "**Run the leakage demonstration and ablations.** E12 (segment split), E14–E20 (channel set, preprocessing, "
        "eyes-open, recording length, client number, within-Mumtaz CV).",
        "**Control for age.** Match groups by age or include age as a covariate and repeat the main comparisons.",
        "**Use real sites.** Add de-identified recordings from a second institution, such as AIIMS, after ethics "
        "approval, so that clients differ in hardware and population.",
        "**Strengthen privacy.** Add secure aggregation so that the server sees only the sum of client updates.",
        "**Address data-set shift.** Study per-site normalisation and domain adaptation, and test whether they help "
        "external transfer without leaking test information."]))
    A(("fig", "fig_roadmap", "Planned path from the present benchmark towards multi-site validation", 0.9))
    return {"meta": META, "abstract": abstract, "keywords": keywords, "blocks": B, "refs": REFS}


def tab_ref(key):
    return "⟦tab:" + key + "⟧"
