import plotly.express as px
import streamlit as st

from common import RESULTS, header, need, table  # noqa: I001

header("Explainability")
st.warning("Everything on this page **explains the model, not the disorder**.")
g, f = table("explain_lr_grouped.parquet"), table("explain_lr_features.parquet")
if need(g, "LR explanations"):
    n_f = int(f["n_folds"].max()) if len(f) else 0
    st.subheader(f"Stable LR coefficients by band × region (sign stable in ≥4/5 folds; {n_f} folds)")
    st.plotly_chart(px.density_heatmap(g, x="region", y="band", z="coef_mean", histfunc="avg", text_auto=".2f",
                                       color_continuous_scale="RdBu_r"), width="stretch")
    st.plotly_chart(px.bar(g, x="region", y="perm_importance", color="band", barmode="group",
                           title="Permutation importance (balanced accuracy drop on test folds), stable features"),
                    width="stretch")
    st.dataframe(f, width="stretch")
png = RESULTS / "explain_eegnet_topomaps.png"
if png.exists():
    st.subheader("EEGNet spatial patterns of the depthwise filters (one fold/seed)")
    st.image(str(png))
occ = table("explain_band_occlusion.parquet")
if need(occ, "band-occlusion results"):
    m = occ.melt(id_vars=["fold", "seed"], value_vars=["delta", "theta", "alpha", "beta"], var_name="band", value_name="drop")
    agg = m.groupby("band", as_index=False).agg(drop=("drop", "mean"), sd=("drop", "std"), n_models=("drop", "size"))
    st.plotly_chart(px.bar(agg, x="band", y="drop", error_y="sd", hover_data=["n_models"],
                           title="Band occlusion: drop in balanced accuracy when a band is removed (mean ± SD over models)"),
                    width="stretch")
