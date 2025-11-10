# app.py
import os
import io
import math
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ---------------------------
# Configuration
# ---------------------------
TOP_K_METRICS = 4
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------
# Helper Functions
# ---------------------------
def startup_message():
    return (
        "Hello! I am your **Vehicle Data Insights Assistant**.\n\n"
        "• Please upload a semicolon-delimited CSV file (`;`).\n"
        "• I will wait for your confirmation before analyzing. Click **Confirm & Analyze** when ready."
    )

def parse_csv_semicolon(upload) -> pd.DataFrame:
    raw = upload.read()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc, errors="replace")
            df = pd.read_csv(io.StringIO(text), sep=";", engine="python", on_bad_lines="skip")
            if df.empty or df.shape[1] < 1:
                continue
            df = coerce_numeric(df)
            time_col = pick_time_column(df)
            if time_col and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)
                except Exception:
                    pass
            return df
        except Exception:
            continue
    raise ValueError("Failed to parse as semicolon-delimited CSV.")

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s, errors="ignore")
    return df

def pick_time_column(df: pd.DataFrame):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
        if df[c].dtype == object:
            try:
                pd.to_datetime(df[c].head(200), errors="raise", infer_datetime_format=True)
                return c
            except Exception:
                continue
    return None

def relevance_score(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    completeness = s.notna().mean()
    variance = np.nanvar(s)
    return 0.7 * completeness + 0.3 * math.log1p(variance)

def select_top_numeric(df: pd.DataFrame, k: int = TOP_K_METRICS) -> list[str]:
    tmp = df.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(tmp[c])]
    scores = [(c, relevance_score(tmp[c])) for c in num_cols]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scores[:k]]

def compute_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = {
            "count": int(s.notna().sum()),
            "missing_pct": float(100 * (1 - s.notna().mean())) if len(s) else 0.0,
            "mean": float(np.nanmean(s)) if s.notna().any() else np.nan,
            "median": float(np.nanmedian(s)) if s.notna().any() else np.nan,
            "std": float(np.nanstd(s)) if s.notna().any() else np.nan,
            "min": float(np.nanmin(s)) if s.notna().any() else np.nan,
            "max": float(np.nanmax(s)) if s.notna().any() else np.nan,
        }
    return out

def structured_summary(stats: dict, limit: int = 4) -> str:
    lines = []
    for i, (col, m) in enumerate(stats.items()):
        if i >= limit:
            break
        def fmt(v): return f"{v:.2f}" if isinstance(v, (int, float)) and not np.isnan(v) else "—"
        lines.append(
            f"- **{col}** — mean: {fmt(m['mean'])}, median: {fmt(m['median'])}, "
            f"min–max: {fmt(m['min'])}–{fmt(m['max'])}, missing: {m['missing_pct']:.1f}%."
        )
    return "\n".join(lines)

def generate_gemini_summary(df: pd.DataFrame, cols: list[str], stats: dict) -> str:
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return "(Gemini not available. Set GOOGLE_API_KEY to enable.)"
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        head_txt = df[cols].head(20).to_markdown(index=False)
        facts_txt = structured_summary(stats, limit=len(cols))
        prompt = f"""
You are a professional, objective Vehicle Data Insights Assistant. Summarize the dataset succinctly.

CONTEXT
- Focus columns: {cols}
- Key descriptive metrics:
{facts_txt}

SAMPLE (first 20 rows):
{head_txt}

REQUIREMENTS
- Tone: formal, neutral, concise.
- Length: 120–170 words.
- Highlight central tendencies, variability, ranges, and data quality (missingness/outliers).
- Only use provided facts; do not speculate.
- Single short paragraph (no bullet points).
"""
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"(Gemini summary failed: {e})"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Vehicle Data Assist1", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Data Assist1 — Semicolon CSV Only")
st.caption("Formal, objective summaries from semicolon-delimited vehicle logs.")

if "df" not in st.session_state:
    st.session_state.df = None
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False

with st.expander("ℹ️ Startup", expanded=True):
    st.markdown(startup_message())

uploaded = st.file_uploader("Upload semicolon-delimited CSV", type=["csv"])
reset = st.button("Reset")

if reset:
    st.session_state.clear()
    st.rerun()

if uploaded and st.session_state.df is None:
    try:
        st.session_state.df = parse_csv_semicolon(uploaded)
        st.info("File parsed successfully. Click **Confirm & Analyze** to proceed.")
    except Exception as e:
        st.error(f"Parsing failed: {e}")

confirm = st.button("✅ Confirm & Analyze", type="primary", disabled=(st.session_state.df is None))
if confirm:
    st.session_state.confirmed = True

if not st.session_state.confirmed:
    st.stop()

# ---------------------------
# Analysis
# ---------------------------
df = st.session_state.df
df_name = uploaded.name if uploaded else "DataFrame"

st.subheader("Dataset Overview")
st.write(f"**Source:** {df_name} • **Rows:** {len(df):,} • **Columns:** {df.shape[1]:,}")
st.dataframe(df.head(15), use_container_width=True)

candidate_cols = select_top_numeric(df, k=TOP_K_METRICS)
if not candidate_cols:
    st.error("No numeric columns detected. Please provide a dataset with numeric measures.")
    st.stop()

st.subheader("Selected Metrics")
st.caption("Automatically selected based on completeness and variance (top 4). You can adjust below.")
user_cols = st.multiselect("Metrics to analyze", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                           default=candidate_cols, max_selections=6)

if not user_cols:
    st.warning("Please select at least one numeric column.")
    st.stop()

stats = compute_stats(df, user_cols)

st.markdown("### Structured Summary (Top Metrics)")
st.markdown(structured_summary(stats, limit=4))

with st.expander("✨ Executive Summary (Gemini — optional)", expanded=False):
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        summary = generate_gemini_summary(df, user_cols[:4], {c: stats[c] for c in user_cols[:4]})
        st.write(summary)
    else:
        st.info("Set GOOGLE_API_KEY in environment to enable Gemini summary.")

ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
report_md = "\n\n".join([
    "# Vehicle Data Assist1 Report",
    f"**Generated:** {ts}",
    f"**Source:** {df_name}",
    "## Top Metrics (Structured)",
    structured_summary({c: stats[c] for c in user_cols}, limit=4),
])
st.download_button("⬇️ Download report (Markdown)", data=report_md.encode("utf-8"),
                   file_name="vehicle_data_assist1_report.md", mime="text/markdown")

st.caption("End of analysis. Upload a new file or adjust metric selection to update results.")
