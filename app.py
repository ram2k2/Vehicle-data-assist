import os
import io
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from google.generativeai.errors import APIError # We still import the Google API error for specific handling

# --- LangChain Imports ---
# We use these high-level components to build the execution logic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------
# Page / Session State Setup
# -----------------------------
st.set_page_config(page_title="Vehicle Data Chat (LCEL)", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Data Chat Assistant — LangChain Agent")
st.caption("Upload a semicolon-delimited CSV and ask questions freely. The LLM acts as an Agent and uses the local Python tool for data computation.")

if "df" not in st.session_state:
    st.session_state.df = None
if "messages_lcel" not in st.session_state:
    # Use a separate message list for this new LCEL application instance
    st.session_state.messages_lcel = []
if "filename" not in st.session_state:
    st.session_state.filename = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

# --- CONSTANTS ---
MODEL_NAME = "gemini-2.5-flash"
# The system prompt guides the LLM to use the tool
AGENT_SYSTEM_PROMPT = """
You are an expert Data Analyst Agent. Your role is to analyze a Pandas DataFrame containing vehicle telemetry data.
The user will ask a question about the data. You MUST use the `execute_pandas_plan` tool to compute the answer.

Follow these steps:
1. Determine the necessary sequence of Pandas operations (filter, aggregate, sort, etc.) needed to answer the user's question.
2. Formulate these operations into a single, valid JSON object that matches the `execute_pandas_plan` tool schema.
3. Call the `execute_pandas_plan` tool with the generated JSON plan.
4. After the tool returns the data (text or table), summarize the result in a concise, friendly, and informative natural language response.
5. If the tool call fails, report the error.
6. **IMPORTANT:** Conclude your response by proactively asking the user a single, relevant, follow-up question based on the data, such as 'Would you like to know the maximum value for [Specific Column Name]?' or 'How did the average [Other Column Name] change over this period?'
"""

# -----------------------------
# Helper Functions (Copied/Modified from app.py)
# -----------------------------

def parse_semicolon_csv(raw: bytes) -> pd.DataFrame:
    """
    STRICT: semicolon is the ONLY delimiter; no auto-detection.
    """
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=";", engine="python", on_bad_lines="skip")
    return df

def coerce_numeric(series: pd.Series) -> pd.Series:
    """Normalize comma-decimals in a column and coerce to numbers."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def case_map(columns: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}

def resolve_col(name: str, columns: List[str]) -> Optional[str]:
    """Resolves a column name (case-insensitive) based on a candidate name."""
    if name in columns:
        return name
    return case_map(columns).get(name.lower())

# --- Core Execution Logic (Now wrapped as a LangChain Tool) ---

@tool
def execute_pandas_plan(plan_json: str) -> str:
    """
    Executes a structured JSON plan against the current Pandas DataFrame
    in the application's memory to perform data analysis operations (filter, 
    aggregate, sort, head, shape, columns).

    The input MUST be a JSON string with the following structure:
    {"plan":[{"op":"<op>", ...args}], "return":"text|table|both"}

    Allowed Operations (ops): shape, columns, aggregate, filter, sort, head
    Allowed Aggregations: mean, median, min, max, std, sum, count, missing_pct
    """
    df = st.session_state.df
    if df is None:
        return "Error: No data frame is loaded. Please upload a CSV first."

    try:
        plan = json.loads(plan_json)
        if not isinstance(plan, dict) or "plan" not in plan:
            return "Execution Error: Invalid plan format (missing 'plan' key or not a dictionary)."
    except json.JSONDecodeError:
        return f"Execution Error: Could not parse input as valid JSON: {plan_json[:100]}..."

    # The plan structure and content is still validated *during execution* for safety,
    # though the Agent is tasked with generating a valid plan from the start.
    from app import validate_plan, execute_plan # Import the existing robust validation/execution logic
    ok, msg, norm_plan = validate_plan(plan, df, max_rows=10) # Reduced max_rows for tool output clarity

    if not ok:
        return f"Execution Error (Validation Failed): {msg}. Original plan: {plan}"

    # Use the existing execution logic
    text_output, table_output = execute_plan(df, norm_plan)
    
    # Store the resulting dataframe in session state for later display by Streamlit
    if isinstance(table_output, pd.DataFrame) and not table_output.empty:
        st.session_state["_last_result_df"] = table_output
    else:
        st.session_state["_last_result_df"] = None

    # Return the text output for the LLM to narrate
    return text_output + f"\n\n(Computed on {len(df):,} rows. Columns used: {df.columns.tolist()})"

# -----------------------------
# Agent Initialization (LCEL)
# -----------------------------

def initialize_agent():
    """Initializes the LLM and the Agent Executor."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("LLM Agent disabled: Please set the GOOGLE_API_KEY environment variable.")
            return

        # 1. Initialize the Chat Model
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key,
            temperature=0.0, # Agents should be deterministic
        )

        # 2. Define the Agent Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        # 3. Define the Tools
        tools = [execute_pandas_plan] # The only tool available is our data executor

        # 4. Create the Agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        # 5. Create the Executor
        st.session_state.agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=False, # Set to True for detailed log output
            handle_parsing_errors=True
        )
        st.success(f"Agent initialized using {MODEL_NAME}.")

    except APIError as e:
        st.error(f"Gemini API Error during initialization: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during agent setup: {e}")

# -----------------------------
# Upload & Initialization Handler
# -----------------------------
uploaded = st.file_uploader("Upload a **semicolon-delimited** CSV", type=["csv"])
if uploaded is not None:
    try:
        raw = uploaded.read()
        raw_df = parse_semicolon_csv(raw)
        st.session_state.df = raw_df.copy()
        st.session_state.filename = uploaded.name
        
        st.success(f"Loaded {uploaded.name} with {len(raw_df):,} rows and {raw_df.shape[1]} columns. Initializing Agent...")
        
        # Display headers for user reference
        with st.expander("View available column headers"):
            st.write(", ".join(raw_df.columns.tolist()))

        # Re-initialize the agent whenever new data is loaded
        initialize_agent()
        
        st.session_state.messages_lcel.append({
            "role": "assistant",
            "content": (
                "CSV loaded. Ask anything (e.g., 'top 5 by vehicle_speed_kmh', 'mean of engine_rpm', 'filter engine_temp_c > 95')."
            )
        })
    except Exception as e:
        st.error(f"Failed to parse CSV or initialize: {e}")


# -----------------------------
# Chat history
# -----------------------------
for m in st.session_state.messages_lcel:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "_df" in m and isinstance(m["_df"], pd.DataFrame):
            st.dataframe(m["_df"], use_container_width=True)

# -----------------------------
# Chat input
# -----------------------------
q = st.chat_input("Ask about your data…", disabled=(st.session_state.df is None or st.session_state.agent_executor is None))
if q:
    st.session_state.messages_lcel.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)

    # Convert Streamlit messages history to LangChain format for context
    langchain_history = []
    for msg in st.session_state.messages_lcel:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            # Only include text content in history for simplicity
            langchain_history.append(SystemMessage(content=msg["content"]))

    if st.session_state.agent_executor:
        with st.spinner("Agent planning and executing computation..."):
            try:
                # Run the Agent
                # The agent will call the execute_pandas_plan tool internally
                response = st.session_state.agent_executor.invoke({
                    "input": q, 
                    "chat_history": langchain_history[:-1] # Exclude the current user message from history
                })
                
                final_answer = response["output"]
                result_df = st.session_state.get("_last_result_df") # Retrieve the DataFrame saved by the tool
                
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                    if isinstance(result_df, pd.DataFrame):
                        st.dataframe(result_df, use_container_width=True)

                record = {"role": "assistant", "content": final_answer}
                if isinstance(result_df, pd.DataFrame):
                    record["_df"] = result_df
                st.session_state.messages_lcel.append(record)

            except Exception as e:
                err = f"Sorry, the Agent encountered a critical error: {e}"
                with st.chat_message("assistant"): st.markdown(err)
                st.session_state.messages_lcel.append({"role": "assistant", "content": err})

    else:
        msg = "LLM Agent is not initialized. Please ensure your `GOOGLE_API_KEY` is set and data is loaded."
        with st.chat_message("assistant"): st.markdown(msg)
        st.session_state.messages_lcel.append({"role": "assistant", "content": msg})