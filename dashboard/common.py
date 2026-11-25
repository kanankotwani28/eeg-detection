"""Shared helpers for the read-only dashboard. Reads ONLY artifacts/results/* (produced by src/report/export.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paths import RESULTS  # noqa: E402

NOT_DIAGNOSTIC = ("**Research benchmark — not a diagnostic tool.** Nothing on this dashboard is a clinical test, "
                  "and no EEG feature shown is a biomarker of depression.")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    p = RESULTS / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def js(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def header(title: str):
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.warning(NOT_DIAGNOSTIC)
    if js("summary.json").get("synthetic"):
        st.error("SYNTHETIC DATA — these results come from generated test data and say nothing about TDBRAIN, "
                 "Mumtaz or depression.")


def need(df: pd.DataFrame, what: str) -> bool:
    if df is None or len(df) == 0:
        st.info(f"No {what} yet. Run the experiments and `make report` (export).")
        return False
    return True


def ci_cols(df: pd.DataFrame, m: str):
    return df[m], df[m + "_hi"] - df[m], df[m] - df[m + "_lo"]


def label(df: pd.DataFrame) -> pd.Series:
    v = df["variant"].where(df["variant"] != "default", "")
    e = df["eps"].where(df["eps"] != "inf", "")
    return (df["experiment"] + " " + v + " " + df["model"] + " " + ("ε=" + e).where(e != "", "")).str.replace("  ", " ")
