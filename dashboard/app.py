"""Overview page. Run with `make dashboard` (streamlit run dashboard/app.py). Read-only, local, no uploads."""
import plotly.graph_objects as go
import streamlit as st

from common import header, js, table  # noqa: I001

header("Federated, privacy-preserving EEG classification of MDD on TDBRAIN")
st.markdown("""
**Study question.** On TDBRAIN, under subject-level evaluation, how much accuracy do EEG-based MDD classifiers lose
when trained federated and with participant-level differential privacy? How much privacy does that buy against a
membership-inference attack (LiRA)? How do the models transfer to an independent dataset (Mumtaz)?
""")
c = js("cohort.json")
fc = c.get("flowchart", {})
if fc:
    st.subheader("Cohort flow")
    stages = ["participants", "first_session", "with_restEC", "labelled_not_blinded"]
    labels_ = ["Participants", "First session", "With restEC", "Labelled (not blinded/UNKNOWN)"]
    fig = go.Figure(go.Funnel(y=labels_ + ["T1 (MDD vs HC)", "T2 (MDD vs other patients)"],
                              x=[fc[s] for s in stages] + [fc["T1_final"], fc["T2_final"]], textinfo="value"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Dropped: REPLICATION/blinded {c.get('dropped_replication_blinded')}, UNKNOWN {c.get('dropped_unknown')} "
               f"(UNKNOWN does not mean healthy), outside age range {c.get('outside_age_range')}.")
st.subheader("Datasets")
ext = table("external.parquet")
qm = js("qc_mumtaz_EC.json")
st.table({
    "Dataset": ["TDBRAIN", "Mumtaz"],
    "Role": ["Training and subject-level CV", "External test, evaluated once (phase 8)"],
    "Participants used": [str(fc.get("T1_final", "—")) + " (T1) / " + str(fc.get("T2_final", "—")) + " (T2)",
                          str(qm.get("n_participants", "—"))],
    "Channels": ["19 shared 10-20 positions (from 26)", "19"],
    "Access": ["Brainclinics (ORCID + data-use agreement); never redistributed", "figshare, doi 10.6084/m9.figshare.4244171"],
})
st.caption("Use the sidebar to open the other pages.")
