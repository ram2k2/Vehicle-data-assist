# app.py
import os
import streamlit as st
import pandas as pd

from vehicle_data_assist1 import VehicleDataAssist1, AgentConfig

st.set_page_config(page_title="Vehicle Data Assist1 (semicolon CSV)", page_icon="🚗", layout="wide")

cfg = AgentConfig(
    top_k_metrics=4,
    use_gemini=bool(os.getenv("GOOGLE_API_KEY")),
    gemini_api_key=os.getenv("GOOGLE_API_KEY"),
    gemini_model="gemini-1.5-flash",
)
agent = VehicleDataAssist1(cfg)

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False

st.title("🚗 Vehicle Data Assist1 — Semicolon CSV Only")
st.caption("Formal, objective summaries from semicolon-delimited CSV logs.")

with st.expander("ℹ️ Startup", expanded=True):
    st.markdown(agent.startup_message())

# Controls
c1, c2 = st.columns([1, 1])
with c1:
    uploaded = st.file_uploader("Upload semicolon-delimited CSV", type=["csv"])
with c2:
    reset = st.button("Reset")

if reset:
    st.session_state.clear()
    st.rerun()

if uploaded is not None and st.session_state.df is None:
    try:
        st.session_state.df = agent.parse_csv_semicolon(uploaded)
        st.info("File parsed successfully. Click **Confirm & Analyze** to proceed.")
    except Exception as e:
        st.error(f"Parsing failed: {e}")

confirm = st.button("✅ Confirm & Analyze", type="primary", disabled=(st.session_state.df is None))
if confirm:
    st.session_state.confirmed = True

if not st.session_state.confirmed:
    st.stop()

# ---------------- Analysis ----------------
df = st.session_state.df
df_name = uploaded.name if uploaded else "DataFrame"

st.subheader("Dataset Overview")
st.write(f"**Source:** {df_name} • **Rows:** {len(df):,} • **Columns:** {df.shape[1]:,}")
st.dataframe(df.head(15), use_container_width=True)

candidate_cols = agent.select_numeric_columns(df, k=cfg.top_k_metrics)
if not candidate_cols:
    st.error("No numeric columns detected. Please provide a dataset with numeric measures.")
    st.stop()

st.subheader("Selected Metrics")
st.caption("Automatically selected by completeness and variance (top 4). You can adjust below.")
user_cols = st.multiselect(
    "Metrics to analyze",
    options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
    default=candidate_cols,
    max_selections=6
)
if not user_cols:
    st.warning("Please select at least one numeric column to continue.")
    st.stop()

stats = agent.compute_stats(df, user_cols)

st.markdown("### Structured Summary (Top Metrics)")
st.markdown(agent.structured_summary(stats, limit=4))

# Optional Gemini executive summary (no visuals)
with st.expander("✨ Executive Summary (Gemini — optional)", expanded=False):
    if cfg.use_gemini:
        text = agent.polished_summary_with_gemini(df, user_cols[:4], {c: stats[c] for c in user_cols[:4]})
        st.write(text)
    else:
        st.info("Set GOOGLE_API_KEY in the environment to enable Gemini.")

# Download (Markdown)
import datetime as dt
ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
md = [
    "# Vehicle Data Assist1 Report",
    f"**Generated:** {ts}",
    f"**Source:** {df_name}",
    "## Top Metrics (Structured)",
    agent.structured_summary({c: stats[c] for c in user_cols}, limit=4),
]
report_md = "\n\n".join(md)
st.download_button(
    "⬇️ Download report (Markdown)",
    data=report_md.encode("utf-8"),
    file_name="vehicle_data_assist1_report.md",
    mime="text/markdown",
)

st.caption("End of analysis. Upload a new file or adjust metric selection to update results.")