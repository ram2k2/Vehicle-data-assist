import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"
REQUIRED_COLUMNS = [
    "Total distance (km)",
    "Fuel efficiency",
    "High voltage battery State of Health (SOH).",
    "Current vehicle speed."
]

@st.cache_resource
def create_agent(df: pd.DataFrame, suffix: str):
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
        agent_kwargs={"suffix": suffix}
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
        df = pd.read_csv(StringIO(data), sep=';')
        st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        st.sidebar.dataframe(df.head(), use_container_width=True)

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

        if missing:
            # Fallback prompt if required columns are missing
            fallback_suffix = """
You are a professional Vehicle Data Analyst. Your job is to analyze vehicle data and provide structured, actionable insights.
Always communicate in a clear, professional, and user-friendly tone.
If the expected columns (`Total distance (km)`, `Fuel efficiency`, `High voltage battery State of Health (SOH).`, `Current vehicle speed.`) are missing, automatically identify the most relevant numeric columns that reflect vehicle performance, health, or efficiency.
Use pandas to clean and convert those columns to numeric format (errors='coerce'), drop rows with missing or invalid values, and compute meaningful metrics such as average values, latest readings, or trends.
Format the output as Markdown:

**🔍 Vehicle Data Summary**
1. **Metric 1**: ...
2. **Metric 2**: ...
3. **Metric 3**: ...
4. **Metric 4**: ...

Note: Provide a brief insight based on the selected metrics.
Always run Python code to generate this summary.
"""
            agent = create_agent(df, fallback_suffix)
        else:
            # Standard prompt when required columns are present
            standard_suffix = """
You are a professional Vehicle Data Analyst. Your job is to analyze vehicle data and provide structured, actionable insights.
You MUST use Python code with pandas to answer questions. DO NOT use df.describe(), df.info(), or generic summaries.
Always communicate in a clear, professional, and user-friendly tone. Avoid technical terms such as "DataFrame", "pandas", "dataset structure", or "data object" unless explicitly asked by the user.
Instead, use natural phrases like:
- your uploaded file
- your data
- the vehicle data
- the file contains...

When asked for a summary, follow this exact protocol:
1. Convert `Total distance (km)`, `Fuel efficiency`, `High voltage battery State of Health (SOH).`, and `Current vehicle speed.` to numeric.
2. Drop rows with missing or invalid values.
3. Calculate:
   - Total Distance Traveled: last - first value of `Total distance (km)`
   - Average Fuel Efficiency
   - Latest Battery SOH
   - Average Vehicle Speed
4. Format the output as Markdown:

**🔍 Vehicle Data Summary**
1. **Total Distance Traveled**: ... km
2. **Average Fuel Efficiency**: ... km/l
3. **Latest Battery SOH**: ...%
4. **Average Vehicle Speed**: ... km/h

Note: Provide a brief insight.
Always run Python code to generate this summary.
"""
            agent = create_agent(df, standard_suffix)

    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        df = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if df is not None and agent is not None:
    # --- Suggested Prompts Section ---
    st.markdown("### 💡 Suggested Prompts")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Comprehensive Summary"):
            st.session_state.suggested_prompt = "Give me a comprehensive summary"

    with col2:
        if st.button("Average Fuel Efficiency"):
            st.session_state.suggested_prompt = "What is the average fuel efficiency?"

    with col3:
        if st.button("Battery SOH Trend"):
            st.session_state.suggested_prompt = "Show battery SOH trend"

    # --- Chat Input or Suggested Prompt ---
    user_input = st.chat_input("Ask about your data (e.g., 'Give me a comprehensive summary')")
    prompt = user_input or st.session_state.get("suggested_prompt", None)

    if prompt:
        st.session_state.suggested_prompt = None  # Clear after use

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
