import pandas as pd
import plotly.express as px
import streamlit as st

from common import header, js, need, table  # noqa: I001

header("Cohort and QC")
c, qc = js("cohort.json"), js("qc.json")
if c:
    rows = [{"task": t, "class": k, "n": v["n"], "age_mean": v["age_mean"], "age_sd": v["age_sd"], "comorbid": v["comorbid"],
             **{f"sex={s}": n for s, n in v["sex_raw_counts"].items()}} for t in ("T1", "T2") for k, v in c[t].items()]
    st.subheader("Class counts, age and sex (raw codes from participants.tsv)")
    st.dataframe(pd.DataFrame(rows), width="stretch")
    st.plotly_chart(px.bar(pd.DataFrame(rows), x="class", y="n", color="task", barmode="group", text="n",
                           title="Participants per class"), width="stretch")
if qc:
    st.subheader(f"Epochs kept per person (n = {qc['n_participants']} participants, {qc['n_excluded']} excluded)")
    k = pd.Series(qc["epochs_kept_per_person"], name="epochs kept")
    st.plotly_chart(px.histogram(k, nbins=30, labels={"value": "clean 4-s epochs kept"}), width="stretch")
    st.subheader("Epoch rejection rate per person")
    r = pd.Series(qc["rejection_rate_per_person"], name="rejection rate")
    st.plotly_chart(px.histogram(r, nbins=30), width="stretch")
    st.write(f"Overall epoch rejection: {qc['epoch_rejection_rate']:.1%}. Recordings with within-recording "
             f"bad-channel interpolation: {qc['n_recordings_with_interpolation']}. Eyes-closed alpha check "
             f"(O1/O2 > Fp1/Fp2): {qc['psd_check']['fraction_EC_alpha_O_gt_Fp']:.0%} of participants.")
psd = table("psd_tdbrain_restEC.parquet")
if need(psd, "PSD"):
    g = psd.groupby(["group", "freq"], as_index=False).agg(mean=("log10_power", "mean"), sd=("log10_power", "std"),
                                                          n=("participant_id", "nunique"))
    g["group"] = g["group"] + " (n=" + g["n"].astype(str) + ")"
    st.subheader("Group-mean PSD (log10 µV²/Hz, mean over 19 channels)")
    st.plotly_chart(px.line(g, x="freq", y="mean", color="group", error_y=None), width="stretch")
