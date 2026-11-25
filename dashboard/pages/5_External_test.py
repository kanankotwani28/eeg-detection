import plotly.graph_objects as go
import streamlit as st

from common import header, need, table  # noqa: I001

header("External test (TDBRAIN → Mumtaz)")
ext, s = table("external.parquet"), table("summary_table.parquet")
REF = {"LR-central": ("E02", "default", "logreg", "inf"), "EEGNet-central": ("E04", "T1", "eegnet", "inf"),
       "LR-FedAvg": ("E06", "iid", "logreg", "inf"), "EEGNet-FedAvg": ("E06", "iid", "eegnet", "inf"),
       "LR-FedAvg-DP-eps8": ("E09", "default", "logreg", "8"), "EEGNet-FedAvg-DP-eps8": ("E09", "default", "eegnet", "8")}
e = ext[ext["kind"] == "external"] if len(ext) else ext
if need(e, "external-test results (E11)"):
    fig = go.Figure()
    for src in e["source"]:
        r = e[e["source"] == src].iloc[0]
        ref = REF.get(src)
        if ref is not None and len(s):
            q = s[(s["experiment"] == ref[0]) & (s["variant"] == ref[1]) & (s["model"] == ref[2]) & (s["eps"] == ref[3])]
            if len(q):
                q = q.iloc[0]
                fig.add_trace(go.Scatter(x=[q["balanced_accuracy"]], y=[src], mode="markers", marker_symbol="circle",
                                         name="TDBRAIN CV", legendgroup="cv", showlegend=src == e["source"].iloc[0],
                                         error_x=dict(type="data", array=[q["balanced_accuracy_hi"] - q["balanced_accuracy"]],
                                                      arrayminus=[q["balanced_accuracy"] - q["balanced_accuracy_lo"]])))
        fig.add_trace(go.Scatter(x=[r["balanced_accuracy"]], y=[src], mode="markers", marker_symbol="diamond",
                                 name=f"Mumtaz (n={int(r['n'])})", legendgroup="mz", showlegend=src == e["source"].iloc[0],
                                 error_x=dict(type="data", array=[r["balanced_accuracy_hi"] - r["balanced_accuracy"]],
                                              arrayminus=[r["balanced_accuracy"] - r["balanced_accuracy_lo"]])))
    fig.add_vline(0.5, line_dash="dash", line_color="grey")
    fig.update_layout(title="TDBRAIN CV vs Mumtaz (balanced accuracy, 95% CI); the gap is the finding",
                      xaxis_title="balanced accuracy")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(e.drop(columns=[c for c in ("mode",) if c in e]), width="stretch")
p = ext[ext["kind"] == "probe"] if len(ext) else ext
if need(p, "dataset-identity probe (E13)"):
    r = p.iloc[0]
    st.metric("Dataset-identity probe AUC (TDBRAIN-HC vs Mumtaz-HC)", f"{r['auc']:.3f}",
              help=f"95% CI [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]; n = {int(r['n_tdbrain_hc'])} + {int(r['n_mumtaz_hc'])}")
    st.caption(f"95% CI [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}], n = {int(r['n_tdbrain_hc'])} TDBRAIN-HC + "
               f"{int(r['n_mumtaz_hc'])} Mumtaz-HC. A high AUC means the datasets are separable without any diagnosis signal.")
if len(s):
    st.subheader("Segment-split leakage demo (E12, deliberately wrong) vs subject split (E04)")
    d = s[(s["experiment"] == "E12") | ((s["experiment"] == "E04") & (s["variant"] == "T1"))]
    st.dataframe(d[["experiment", "model", "n_participants", "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi", "auc"]],
                 width="stretch")
mcv = ext[ext["kind"] == "mumtaz_cv"] if len(ext) else ext
if len(mcv):
    st.subheader("E20: within-Mumtaz subject-level vs segment-level CV")
    st.dataframe(mcv[["model", "mode", "participant_ba", "participant_auc", "segment_ba", "segment_auc"]], width="stretch")
