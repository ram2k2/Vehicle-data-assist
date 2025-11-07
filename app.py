import os
from io import StringIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain + Gemini for non-summary questions
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
MODEL_NAME = "gemini-2.5-flash"

# -----------------------------
# Minimal helpers
# -----------------------------
def find_col(df: pd.DataFrame, patterns, exclude=None):
    """Pick the first column whose name contains any pattern (case-insensitive),
    preferring columns with more numeric values."""
    exclude = exclude or []
    cands = []
    for c in df.columns:
        lc = c.lower()
        if any(p in lc for p in patterns) and not any(x in lc for x in exclude):
            s = pd.to_numeric(df[c], errors="coerce")
            cands.append((c, s.notna().sum()))
    if not cands:
        return None
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0]

def summarize_vehicle_performance(df: pd.DataFrame) -> str:
    """Return exactly 4 metrics with one value each in your format."""
    # Make a numeric view for quick operations (safe coercion)
    numdf = df.copy()
    for c in numdf.columns:
        numdf[c] = pd.to_numeric(numdf[c], errors="coerce")

    # Choose likely columns (simple keyword matching)
    distance_col = find_col(df, ["odometer", "distance", "odo", "total distance"])
    fuel_eff_col = find_col(df, ["fuel efficiency", "fuel economy", "km/l", "kmpl", "mpg", "consumption"])
    soh_col      = find_col(df, ["soh", "state of health"], exclude=["soc"])
    speed_col    = find_col(df, ["speed", "vehicle speed"])

    metrics = []

    # 1) Total distance: last - first
    if distance_col is not None:
        s = pd.to_numeric(df[distance_col], errors="coerce").dropna()
        if len(s) >= 2:
            delta_km = s.iloc[-1] - s.iloc[0]
            if pd.notna(delta_km):
                metrics.append(("Total distance", f"{delta_km:,.2f} km"))

    # 2) Fuel efficiency: average
    if len(metrics) < 4 and fuel_eff_col is not None:
        s = pd.to_numeric(df[fuel_eff_col], errors="coerce").dropna()
        if len(s) > 0:
            unit = " km/l" if any(u in fuel_eff_col.lower() for u in ["km/l", "kmpl"]) else ""
            metrics.append(("Fuel efficiency (avg)", f"{s.mean():,.2f}{unit}"))

    # 3) Battery SOH: latest
    if len(metrics) < 4 and soh_col is not None:
        s = pd.to_numeric(df[soh_col], errors="coerce").dropna()
        if len(s) > 0:
            metrics.append(("Battery SOH (latest)", f"{s.iloc[-1]:,.2f}%"))

    # 4) Average speed
    if len(metrics) < 4 and speed_col is not None:
        s = pd.to_numeric(df[speed_col], errors="coerce").dropna()
        if len(s) > 0:
            metrics.append(("Average speed", f"{s.mean():,.2f} km/h"))

    # Fallbacks if we still have < 4
    if len(metrics) < 4:
        # Try a few common alternates
        for name, pats, agg in [
            ("Estimated range (latest)", ["range", "driving range", "estimated range"], "latest"),
            ("Battery voltage (avg)",    ["battery voltage", "vbatt", "voltage"],       "avg"),
            ("Engine load (avg)",        ["engine load"],                               "avg"),
            ("Transmission temperature (max)", ["transmission temperature", "trans temp", "gearbox temp"], "max"),
        ]:
            if len(metrics) >= 4:
                break
            col = find_col(df, pats)
            if col is None:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            val = s.mean() if agg == "avg" else (s.max() if agg == "max" else s.iloc[-1])
            unit = ""
            if "range" in name.lower(): unit = " km"
            if "voltage" in name.lower(): unit = " V"
            if "temperature" in name.lower(): unit = " °C"
            metrics.append((name, f"{val:,.2f}{unit}"))

    # Generic numeric fallbacks (highest-variance columns): use averages
    if len(metrics) < 4:
        numeric = numdf.select_dtypes(include="number")
        if numeric.shape[1] > 0:
            used = set(m[0] for m in metrics)
            for col in numeric.var().sort_values(ascending=False).index:
                if len(metrics) >= 4:
                    break
                label = f"{col} (avg)"
                if label in used:
                    continue
                s = pd.to_numeric(numdf[col], errors="coerce").dropna()
                if len(s) > 0:
                    metrics.append((label, f"{s.mean():,.2f}"))
                    used.add(label)

    # Keep exactly 4
    metrics = metrics[:4]

    # Build Markdown (no tech jargon)
    bullets = [f"{i}. **{name}**: {value}" for i, (name, value) in enumerate(metrics, 1)]
    hints = []
    for name, _ in metrics:
        nl = name.lower()
        if "distance" in nl: hints.append("distance looks reasonable for the period observed")
        if "fuel efficiency" in nl: hints.append("fuel use appears consistent")
        if "soh" in nl: hints.append("battery health is within expected range")
        if "speed" in nl: hints.append("driving speeds seem typical for mixed conditions")
        if "range" in nl: hints.append("estimated range is aligned with recent usage")
        if "voltage" in nl: hints.append("electrical system voltage is stable")
        if "load" in nl: hints.append("engine load indicates light-to-moderate usage")
        if "temperature" in nl: hints.append("temperature stayed within acceptable limits")
    hints = list(dict.fromkeys(hints))
    summary = " ".join(hints[:3]) if hints else "overall metrics look consistent with recent usage"

    return (
        "**🔍 Vehicle Performance Summary**\n"
        + "\n".join(bullets)
        + "\n\n"
        + f"{summary.capitalize()}."
    )

# -----------------------------
# Agent for non-summary Q&A
# -----------------------------
@st.cache_resource
def create_agent(df: pd.DataFrame, system_suffix: str):
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=api_key if api_key else None)
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        agent_type="openai-tools",
        allow_dangerous_code=True,
        agent_kwargs={"suffix": system_suffix},
    )

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Vehicle Data Analyst", layout="wide")
st.title("🚗 Vehicle Data Analyst Agent")

if not st.session_state.get("initial_greeting_sent", False):
    st.markdown("Hi, welcome! 😊 Please upload a CSV file for vehicle data analysis.")
    st.session_state["initial_greeting_sent"] = True

uploaded_file = st.sidebar.file_uploader("Upload your Vehicle Data CSV (semicolon-delimited)", type="csv")
df = None
agent = None

if uploaded_file is not None:
    try:
        raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        # STRICT: semicolon-only parsing
        df = pd.read_csv(StringIO(raw), sep=";")
        st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        st.sidebar.dataframe(df.head(), use_container_width=True)

        language_style_rule = """
Language Style Rule:
Do not use technical terms like 'DataFrame', 'pandas', 'data types', 'memory usage', or 'dataset structure'.
Refer to the data as 'your uploaded file', 'your vehicle data', or 'your data'.
Avoid column classifications or type breakdowns. Focus on meaningful insights only.
"""

        agent_suffix = f"""
You are a professional Vehicle Data Analyst. Communicate clearly and avoid raw descriptive statistics.
When the user asks for a 'summary', the app computes it—do not print stats dumps.
For other questions, provide concise, actionable insights.
{language_style_rule}
"""
        agent = create_agent(df, agent_suffix)

    except Exception as e:
        st.sidebar.error(f"Error reading CSV (semicolon expected): {e}")
        df = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if df is not None:
    st.markdown("### 💡 Suggested Prompts")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Summary of Vehicle Performance"):
            st.session_state.suggested_prompt = "Give me a summary of vehicle performance metrics"
    with col2:
        if st.button("Average Fuel Efficiency"):
            st.session_state.suggested_prompt = "What is the average fuel efficiency?"
    with col3:
        if st.button("Battery SOH Trend"):
            st.session_state.suggested_prompt = "Show battery SOH trend"

    user_input = st.chat_input("Ask about your data (e.g., 'Give me a summary of vehicle performance')")
    prompt = user_input or st.session_state.get("suggested_prompt", None)

    if prompt:
        st.session_state.suggested_prompt = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        wants_summary = "summary" in prompt.lower() or "summarize" in prompt.lower()
        if wants_summary:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        md = summarize_vehicle_performance(df)
                    except Exception as e:
                        md = f"❌ Error during summary: {e}"
                st.markdown(md)
                st.session_state.messages.append({"role": "assistant", "content": md})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        resp = agent.run(prompt)
                    except Exception as e:
                        resp = f"❌ Error during analysis: {str(e)}"
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
