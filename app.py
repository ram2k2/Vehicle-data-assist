# app.py
import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vehicle Data Chat", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Data Chat Assistant")
st.caption("Hi! Please upload a CSV file for Vehicle Data Analysis.")

# -----------------------------
# Session state
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = None

# -----------------------------
# Constants & helpers
# -----------------------------
PRIMARY_COLS = [
    "Total distance (km)",
    "Fuel efficiency",
    "High voltage battery State of Health (SOH).",
    "Current vehicle speed.",
]
INVALID_TOKENS = {"", "NA", "NV", None}
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def footer_text(filename: str | None) -> str:
    name = filename if filename else "no file"
    return f"\n\n*data extracted from ({name})*"

def parse_semicolon_csv_strict(raw: bytes) -> pd.DataFrame:
    """Strict semicolon parsing; no delimiter auto-detection."""
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=";", engine="python", on_bad_lines="skip")
    return df

def first_numeric_subvalue(cell: str | float | int) -> float:
    """
    Within a single cell (already split by ';'), cells may contain comma-separated sub-values
    like '13.525000,13.375000'. Pick the FIRST numeric token deterministically.
    """
    if cell is None:
        return np.nan
    s = str(cell).strip()
    if s in INVALID_TOKENS:
        return np.nan
    # split by comma INSIDE the cell (not the CSV delimiter)
    parts = [p.strip() for p in s.split(",")]
    for p in parts:
        if p in INVALID_TOKENS:
            continue
        # Try plain float, else handle comma-as-decimal
        try:
            return float(p)
        except Exception:
            if "," in p and "." not in p:  # e.g., "13,5"
                try:
                    return float(p.replace(",", "."))
                except Exception:
                    pass
        # Fallback: extract numeric with regex
        m = NUM_RE.search(p)
        if m:
            try:
                return float(m.group())
            except Exception:
                pass
    return np.nan

def clean_primary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PRIMARY_COLS:
        if col in out.columns:
            out[col] = out[col].apply(first_numeric_subvalue)
    # Drop columns that end up entirely empty (ignored as per instruction)
    for col in PRIMARY_COLS:
        if col in out.columns and out[col].notna().sum() == 0:
            out.drop(columns=[col], inplace=True)
    return out

def summarize_metrics(df: pd.DataFrame, filename: str | None) -> str:
    # Ensure required columns exist
    missing = [c for c in PRIMARY_COLS if c not in df.columns]
    if missing:
        return (
            "**I couldn't find these required headers:** "
            + ", ".join(f"`{c}`" for c in missing)
            + ".\nPlease share the exact column names (case-sensitive) or provide a mapping."
            + footer_text(filename)
        )

    s_distance = df["Total distance (km)"].dropna()
    s_fueleff = df["Fuel efficiency"].dropna()
    s_soh     = df["High voltage battery State of Health (SOH)."].dropna()
    s_speed   = df["Current vehicle speed."].dropna()

    # Calculations (as specified)
    total_distance = float(s_distance.iloc[-1] - s_distance.iloc[0]) if len(s_distance) >= 2 else np.nan
    avg_fe        = float(s_fueleff.mean()) if len(s_fueleff) else np.nan
    latest_soh    = float(s_soh.iloc[-1]) if len(s_soh) else np.nan
    avg_speed     = float(s_speed.mean()) if len(s_speed) else np.nan

    def fmt(x, d=3):
        return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{d}f}"

    body = (
        f"**Total Distance:** {fmt(total_distance)} km\n"
        f"**Average Fuel Efficiency:** {fmt(avg_fe)} (units per file)\n"
        f"**Latest Battery SOH:** {fmt(latest_soh, 0)} %\n"
        f"**Average Vehicle Speed:** {fmt(avg_speed)} (units per file)"
    )

    # If any of the four columns contained inner commas in the ORIGINAL data, note the assumption
    note = ""
    try:
        if st.session_state.raw_df is not None:
            for col in PRIMARY_COLS:
                if col in st.session_state.raw_df.columns:
                    if st.session_state.raw_df[col].astype(str).str.contains(",", regex=False).any():
                        note = ("\n\n_Note: Cells with comma‑separated values were parsed using the **first** "
                                "numeric value. If you prefer a different rule (e.g., average/second value), "
                                "tell me and I’ll recalculate._")
                        break
    except Exception:
        pass

    return body + note + footer_text(filename)

# -----------------------------
# Upload area (only user CSV; no Test.csv)
# -----------------------------
uploaded_file = st.file_uploader("Upload a **semicolon-delimited** CSV", type=["csv"])

if uploaded_file:
    try:
        raw = uploaded_file.read()
        raw_df = parse_semicolon_csv_strict(raw)
        st.session_state.raw_df = raw_df.copy()
        df = clean_primary_columns(raw_df)
        st.session_state.df = df
        st.session_state.filename = uploaded_file.name

        st.success(f"Loaded {uploaded_file.name} with {len(raw_df):,} rows and {raw_df.shape[1]} columns.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "CSV loaded. You can now ask:\n"
                "- `summary` (calculates Total Distance, Avg Fuel Efficiency, Latest SOH, Avg Speed)\n"
                "- `columns` (list headers)\n"
            ) + footer_text(st.session_state.filename)
        })
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

# -----------------------------
# Show chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Disable chat until a file is uploaded
query = st.chat_input("Ask about your data…", disabled=(st.session_state.df is None))

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.df is None:
        reply = "Please upload a semicolon-delimited CSV first." + footer_text(None)
    else:
        df = st.session_state.df
        cols = list(st.session_state.raw_df.columns.astype(str)) if st.session_state.raw_df is not None else []
        q = query.strip().lower()

        if "summary" in q:
            reply = summarize_metrics(df, st.session_state.filename)

        elif "columns" in q:
            reply = ("Columns:\n- " + "\n- ".join(cols)) + footer_text(st.session_state.filename)

        else:
            # Minimal fallback (keep it lean)
            reply = (
                "I can help with:\n"
                "- `summary`\n"
                "- `columns`\n"
                "If column names differ from the expected headers, please share the exact names or a mapping."
            ) + footer_text(st.session_state.filename)

    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})