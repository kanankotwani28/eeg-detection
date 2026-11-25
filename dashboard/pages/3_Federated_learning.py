import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import header, label, need, table  # noqa: I001

header("Federated learning")
h = table("fl_history.parquet")
if need(h, "federated round histories"):
    exps = sorted(h["experiment"].unique())
    e = st.selectbox("Experiment", exps, index=exps.index("E06") if "E06" in exps else 0)
    d = h[h["experiment"] == e]
    g = d.groupby(["variant", "model", "eps", "round"], as_index=False).agg(val_ba=("val_ba", "mean"), sd=("val_ba", "std"),
                                                                           n_runs=("val_ba", "size"))
    g["config"] = g["variant"] + " " + g["model"] + " ε=" + g["eps"]
    st.subheader("Server-side validation balanced accuracy per round (mean over folds × seeds)")
    st.plotly_chart(px.line(g, x="round", y="val_ba", color="config", error_y="sd", hover_data=["n_runs"]),
                    width="stretch")
s = table("summary_table.parquet")
if need(s, "results"):
    def dots(d, title):
        d = d.copy()
        d["config"] = label(d) + " (n=" + d["n_participants"].astype(int).astype(str) + ")"
        fig = go.Figure(go.Scatter(x=d["balanced_accuracy"], y=d["config"], mode="markers", error_x=dict(
            type="data", array=d["balanced_accuracy_hi"] - d["balanced_accuracy"],
            arrayminus=d["balanced_accuracy"] - d["balanced_accuracy_lo"])))
        fig.update_layout(title=title, height=60 + 28 * len(d), xaxis_title="balanced accuracy (95% CI)")
        st.plotly_chart(fig, width="stretch")
    dots(s[s["experiment"].isin(["E02", "E04", "E06"]) & s["variant"].isin(["default", "T1", "iid", "label_skew", "iid_dropout"])],
         "Centralised vs FedAvg: IID vs label-skew (and client dropout p=0.2)")
    dots(s[s["experiment"].isin(["E06", "E07", "E08"]) & (s["model"] == "eegnet")], "FedAvg vs FedProx vs FedBN (EEGNet)")
    k = s[s["experiment"] == "E17"]
    if len(k):
        k = k.assign(K=k["variant"].str[1:].astype(int))
        st.plotly_chart(px.line(k.sort_values("K"), x="K", y="balanced_accuracy", markers=True,
                                error_y=k["balanced_accuracy_hi"] - k["balanced_accuracy"],
                                error_y_minus=k["balanced_accuracy"] - k["balanced_accuracy_lo"],
                                title="Number of clients K"), width="stretch")
    loc = s[s["experiment"] == "E05"]
    if len(loc):
        st.subheader("Local-only (E05): mean client vs worst client")
        st.dataframe(loc[["variant", "model", "n_participants", "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi",
                          "worst_balanced_accuracy", "worst_balanced_accuracy_lo", "worst_balanced_accuracy_hi"]],
                     width="stretch")
