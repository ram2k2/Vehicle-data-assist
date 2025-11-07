import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

# Updated to use 'prefix' instead of 'suffix' for stronger instruction enforcement
@st.cache_resource
def create_agent(df: pd.DataFrame, prefix: str):
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=api_key if api_key else None
    )
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        agent_type="openai-tools",
        allow_dangerous_code=True,
        # Using 'prefix' ensures these instructions are prepended and strictly enforced
        agent_kwargs={"prefix": prefix}
    )

# --- Streamlit UI ---
st.set_page_config(page_title="Vehicle Data Analyst", layout="wide")
st.title("🚗 Vehicle Data Analyst Agent")

if not st.session_state.get("initial_greeting_sent", False):
    st.markdown("Hi, welcome! 😊 Please upload a CSV file for vehicle data analysis.")
    st.session_state["initial_greeting_sent"] = True

uploaded_file = st.sidebar.file_uploader("Upload your Vehicle Data CSV", type="csv")
df = None
agent = None

if uploaded_file is not None:
    try:
        data = uploaded_file.getvalue().decode("utf-8")
        # STRICT: semicolon-only parsing
        df = pd.read_csv(StringIO(data), sep=';')
        st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        st.sidebar.dataframe(df.head(), use_container_width=True)

        # Combined and enforced instructions using a strong agent prefix
        agent_prefix = """
You are a professional Vehicle Data Analyst. Your job is to analyze vehicle data and provide structured, actionable insights.
Always communicate in a clear, professional, and user-friendly tone.

--- **MANDATORY RULES FOR ALL RESPONSES** ---
1. You MUST NOT use technical terms like 'DataFrame', 'pandas', 'data types', 'memory usage', 'index', or 'dataset structure'.
2. Instead, refer to the data as 'your uploaded file', 'your vehicle data', or 'your data'.
3. You MUST NOT include column classifications or raw data type breakdowns.
4. **CRITICAL:** You MUST NOT include raw descriptive statistics tables (count/mean/std/min/max/percentiles) in any response, especially for summaries. **DO NOT use pandas functions like df.describe() or df.info().**
5. You MUST NOT show intermediate Python code or steps.

--- **SPECIFIC RULE FOR SUMMARY REQUESTS (e.g., 'summary of vehicle performance')** ---
If the user asks for a summary (or equivalent intent), you MUST:
- Pre-process: Clean and convert all relevant columns to numeric format (errors='coerce') and drop rows with missing or invalid values.
- Metrics: Calculate the following FOUR metrics:
    1. Total distance traveled: Last recorded value minus the first recorded value of 'Total Distance (km)'.
    2. Average speed: The mean of the 'Current Vehicle Speed' column.
    3. Average fuel efficiency: The mean of the 'Fuel Efficiency' column.
    4. Current battery health: The LAST recorded value of 'High Voltage Battery State of Health (SOH)'.
- Output Constraint: You MUST output **ONLY** the final formatted Markdown block and the 2-3 line summary paragraph. Do NOT include any intermediate steps, calculations, or descriptive statistics tables.
- Format the output EXACTLY as Markdown:

**🔍 Vehicle Performance Summary**
1. **Total Distance Traveled**: [Calculated Value from Metric 1]
2. **Average Vehicle Speed**: [Calculated Value from Metric 2]
3. **Average Fuel Efficiency**: [Calculated Value from Metric 3]
4. **Current Battery SOH**: [Calculated Value from Metric 4]

Then provide a short summary paragraph (2–3 lines) describing the vehicle's overall performance based on these metrics.
Always run Python code to generate this summary. Do not guess values—compute them from the uploaded file.
"""

        agent = create_agent(df, agent_prefix)

    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        df = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if df is not None and agent is not None:
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

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    agent_response = agent.run(prompt)
                except Exception as e:
                    agent_response = f"❌ Error during analysis: {str(e)}"

            st.markdown(agent_response)
            st.session_state.messages.append({"role": "assistant", "content": agent_response})
