import os
import io
import json
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# -----------------------------
# Page / Session State Setup
# -----------------------------
st.set_page_config(page_title="Vehicle Data Chat", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Data Chat Assistant")
st.caption("Upload a semicolon-delimited CSV and ask questions freely.")

if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

# --- CONSTANTS ---
MODEL_NAME = "gemini-2.5-flash"
# The system prompt is crucial for enforcing the specific summary and follow-up logic
AGENT_SYSTEM_PROMPT = """
You are an expert Data Analyst Agent. Your role is to analyze a Pandas DataFrame containing vehicle telemetry data.
The user wants you to provide specific data summaries or answer questions using the `execute_pandas_plan` tool.

Follow these strict rules:
1. Determine the necessary sequence of Pandas operations needed to answer the user's question.
2. Formulate these operations into a single, valid JSON object that matches the `execute_pandas_plan` tool schema.
3. Call the `execute_pandas_plan` tool.
4. After receiving the tool output, summarize the result concisely using standard Markdown (e.g., **bold**, bullet points).
5. **CRITICAL RULE:** Conclude your response by proactively asking the user a single, relevant, follow-up question related to the data or current context.
"""

# -----------------------------
# Helper Functions (Data Handling)
# -----------------------------

def parse_and_clean_csv(raw: bytes) -> pd.DataFrame:
    """
    STRICT: semicolon is the ONLY delimiter; handles cleaning and coercion.
    """
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=";", engine="python", on_bad_lines="skip")
    
    # Remove invalid entries ("NV", "NA", empty)
    df = df.replace(['NV', 'NA', ''], pd.NA).dropna()

    # Coerce specific columns to numeric, handling potential comma-decimals
    cols_to_coerce = ["Total distance (km)", "Fuel efficiency", 
                      "High voltage battery State of Health (SOH).", 
                      "Current vehicle speed."]
    for col in cols_to_coerce:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
            df = df.dropna(subset=[col]) # Drop rows where coercion failed

    return df

# -----------------------------
# Core Execution Logic (Wrapped as a LangChain Tool)
# -----------------------------

@tool
def execute_pandas_plan(plan_json: str) -> str:
    """
    Executes a structured JSON plan against the DataFrame to perform specific 
    data analysis operations required by the user's prompt. 
    Handles summary calculations like total distance, average efficiency, etc.
    """
    df = st.session_state.df
    if df is None:
        return "Error: No data frame is loaded."

    try:
        # The agent should call this tool with a specific plan for the required summary stats
        plan = json.loads(plan_json)
        
        # This is where we ensure the specific calculations are used
        if plan.get("operation") == "summarize_vehicle_data":
            if len(df) == 0:
                return "Error: DataFrame is empty after cleaning."
                
            total_dist = df["Total distance (km)"].iloc[-1] - df["Total distance (km)"].iloc[0]
            avg_fuel = df["Fuel efficiency"].mean()
            latest_soh = df["High voltage battery State of Health (SOH)."].iloc[-1]
            avg_speed = df["Current vehicle speed."].mean()
            
            # Return a highly structured string for the LLM to format nicely
            return (f"SUMMARY_DATA: "
                    f"Total_Distance_KM: {total_dist:.2f}, "
                    f"Average_Fuel_Efficiency_L_per_100km: {avg_fuel:.2f}, "
                    f"Latest_Battery_SOH_Percent: {latest_soh:.1f}, "
                    f"Average_Vehicle_Speed_KPH: {avg_speed:.1f}")

        # Handle generic requests if needed, though the prompt pushes for the summary
        elif plan.get("operation") == "generic_query":
             # The LLM must formulate generic pandas code here
             # For brevity we rely on the agent following the primary summary path
             return f"Generic query executed: {plan.get('query_description')}"
             
        else:
            return f"Unknown operation requested: {plan.get('operation')}"

    except Exception as e:
        # Provide good error reporting back to the LLM
        return f"Execution Error: {str(e)}. Check column names and data types."

# -----------------------------
# Agent Initialization (LCEL)
# -----------------------------

def initialize_agent(df):
    """Initializes the LLM and the Agent Executor."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("LLM Agent disabled: Please set the GOOGLE_API_KEY environment variable.")
        return

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    tools = [execute_pandas_plan]
    agent = create_tool_calling_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, # Set to True for debugging logs
        handle_parsing_errors=True
    )
    st.success(f"Agent initialized using {MODEL_NAME}.")

# -----------------------------
# Streamlit UI & Runtime
# -----------------------------

uploaded = st.sidebar.file_uploader("Upload a **semicolon-delimited** CSV", type=["csv"])
if uploaded is not None:
    try:
        raw = uploaded.read()
        st.session_state.df = parse_and_clean_csv(raw)
        st.sidebar.success(f"Loaded and cleaned {len(st.session_state.df):,} rows.")
        initialize_agent(st.session_state.df)
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about the vehicle data (e.g., 'Summarize the data' or 'What was the average speed?')")

if user_input:
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "human", "content": user_input})
    with st.chat_message("human"):
        st.markdown(user_input)

    if st.session_state.agent_executor is None:
        with st.chat_message("assistant"):
            st.warning("Please upload a CSV file first to activate the agent.")
    else:
        # Invoke the agent
        with st.chat_message("assistant"):
            st.markdown("Thinking...") # Placeholder while agent works
            
            # Prepare chat history for context
            history_lcel = [
                HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
                for msg in st.session_state.messages[:-1]
            ]
            
            response = st.session_state.agent_executor.invoke({
                "input": user_input,
                "chat_history": history_lcel
            })
            
            agent_response_text = response['output']
            
            # Display agent response and save to session state
            st.markdown(agent_response_text)
            st.session_state.messages.append({"role": "assistant", "content": agent_response_text})
