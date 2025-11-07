import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
MODEL_NAME = "gemini-2.5-flash"
REQUIRED_COLUMNS = [
    "Total distance (km)",
    "Fuel efficiency",
    "High voltage battery State of Health (SOH).",
    "Current vehicle speed."
]

# --- AGENT CREATION ---
@st.cache_resource
def create_agent(df: pd.DataFrame):
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=api_key if api_key else None
    )

    SYSTEM_PROMPT_SUFFIX = (
        "You are an expert **Vehicle Data Analyst** working with data obtained from Data Act access requests. "
        "Your goal is to provide **meaningful, useful, and actionable insights** about vehicle health, efficiency, and usage. "
        "Your responses must be **formal, direct, and authoritative**.\n"
        "--- Mandatory Protocol ---\n"
        "1. **Execution**: You MUST generate and run Python code using pandas to answer questions.\n"
        "2. **Initial Check**: Immediately state the exact row and column count of the loaded DataFrame.\n"
        "3. **Summary Command**: When the user requests a 'summary' or 'comprehensive breakdown', follow these steps:\n"
        "    a. **Data Cleaning**: Locate columns: `Total distance (km)`, `Fuel efficiency`, `High voltage battery State of Health (SOH).`, `Current vehicle speed.`. "
        "Convert to numeric (errors='coerce') and drop rows with missing/invalid data.\n"
        "    b. **Calculations**:\n"
        "        - Total Distance Traveled: final - initial value of `Total distance (km)`\n"
        "        - Average Fuel Efficiency: mean of `Fuel efficiency`\n"
        "        - Latest Battery SOH: last value of `High voltage battery State of Health (SOH).`\n"
        "        - Average Vehicle Speed: mean of `Current vehicle speed.`\n"
        "    c. **Formatted Response**: Present results in Markdown:\n"
        "**🔍 Vehicle Data Summary**\n"
        "1. **Total Distance Traveled**: ... km\n"
        "2. **Average Fuel Efficiency**: ... km/l\n"
        "3. **Latest Battery SOH**: ...%\n"
        "4. **Average Vehicle Speed**: ... km/h\n"
        "Note: Provide a brief insight.\n"
        "---\n"
        "**DO NOT** use df.describe(), df.info(), or generic summaries. You MUST write and execute custom pandas code."
    )

    agent_executor = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        agent_type="openai-tools",
        allow_dangerous_code=True,
        agent_kwargs={"suffix": SYSTEM_PROMPT_SUFFIX}
    )
    return agent_executor

# --- STREAMLIT UI ---
st.set_page_config(page_title="Vehicle Data Analyst", layout="wide")
st.title("🚗 Vehicle Data Analyst Agent")

if not st.session_state.get("initial_greeting_sent", False):
    st.markdown("Hi, welcome! 😊 Please upload a CSV file for vehicle data analysis.")
    st.session_state["initial_greeting_sent"] = True

uploaded_file = st.sidebar.file_uploader("Upload your Vehicle Data CSV", type="csv")
df = None

if uploaded_file is not None:
    try:
        data = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(data), sep=';')
        st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        st.sidebar.dataframe(df.head(), use_container_width=True)

        # Validate required columns
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        df = None

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if df is not None:
    agent = create_agent(df)

    if prompt := st.chat_input("Ask about your data (e.g., 'Give me a comprehensive summary')"):
        if agent is None:
            st.warning("Agent not initialized. Check your GEMINI_API_KEY.")
            st.stop()

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
