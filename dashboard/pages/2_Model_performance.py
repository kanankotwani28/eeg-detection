import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import header, label, need, table  # noqa: I001

header("Model performance")
s = table("summary_table.parquet")
if need(s, "results"):
    exps = sorted(s["experiment"].unique())
    pick = st.multiselect("Experiment ID", exps, default=[e for e in exps if e in ("E01", "E02", "E03", "E04", "E06", "E12", "E16")] or exps)
    d = s[s["experiment"].isin(pick)].copy()
    d["config"] = label(d) + " (n=" + d["n_participants"].astype(int).astype(str) + ")"
    for m in ("balanced_accuracy", "auc"):
        fig = go.Figure(go.Scatter(x=d[m], y=d["config"], mode="markers",
                                   error_x=dict(type="data", array=d[m + "_hi"] - d[m], arrayminus=d[m] - d[m + "_lo"])))
        fig.add_vline(0.5, line_dash="dash", line_color="grey")
        fig.update_layout(height=40 + 26 * len(d), xaxis_title=f"{m} (participant-level, 95% bootstrap CI)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
    st.caption("E12 is a DELIBERATELY WRONG segment-level split. E01 uses shuffled labels and should be ≈ 0.5.")
    roc = table("roc.parquet")
    if need(roc, "ROC curves"):
        roc = roc[roc["experiment"].isin(pick)].copy()
        roc["config"] = label(roc)
        f = px.line(roc, x="fpr", y="tpr", color="config")
        f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        st.subheader("ROC curves (seed-averaged out-of-fold predictions)")
        st.plotly_chart(f, width="stretch")
    cal = table("calibration.parquet")
    if need(cal, "calibration"):
        cal = cal[cal["experiment"].isin(pick)].copy()
        cal["config"] = label(cal)
        f = px.line(cal, x="mean_pred", y="frac_pos", color="config", markers=True, hover_data=["n"])
        f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        st.subheader("Calibration (10 bins; hover shows n per bin)")
        st.plotly_chart(f, width="stretch")
    cm = table("confusion.parquet")
    if need(cm, "confusion matrices"):
        cm = cm[cm["experiment"].isin(pick)].copy()
        cm["config"] = label(cm)
        c = st.selectbox("Confusion matrix for", cm["config"].tolist())
        r = cm[cm["config"] == c].iloc[0]
        st.plotly_chart(px.imshow([[r.tn, r.fp], [r.fn, r.tp]], text_auto=True, x=["pred HC/other", "pred MDD"],
                                  y=["true HC/other", "true MDD"]), width="stretch")
    comps = table("comparisons.parquet")
    if need(comps, "paired comparisons"):
        st.subheader("Paired bootstrap differences (A − B, 95% CI; no p-values)")
        st.dataframe(comps, width="stretch")
