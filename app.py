import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from io import StringIO
from dotenv import load_dotenv

# CORRECT FIX: Import the Pandas agent creator from the specific toolkits path.
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent 

# Load environment variables (useful for local testing)
load_dotenv() 

# --- 1. CONFIGURATION & UI SETUP ---
MODEL_NAME = "gemini-2.5-flash"
st.set_page_config(page_title="Free Vehicle Data Agent", layout="wide")
st.title("🚗 Free Vehicle Data Insights Agent")
st.write("Upload your CSV file to get instant summaries and accurate insights using the Gemini AI model.")

# --- 2. AGENT CREATION FUNCTION ---
# This function is cached to prevent redundant object creation
@st.cache_resource
def create_agent(df: pd.DataFrame):
    """Initializes and returns the LangChain Pandas Dataframe Agent."""
    
    # Check for API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY environment variable is not set. Cannot proceed.")
        return None

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=api_key, temperature=0.0)
    
    # Use create_pandas_dataframe_agent, which correctly accepts the 'df' object
    agent_executor = create_pandas_dataframe_agent(
        llm=llm,
        df=df,  # Pass the DataFrame directly
        verbose=False, 
        agent_type="openai-tools",
        # MANDATORY: Allows the LLM to run Python code for data analysis
        allow_dangerous_code=True, 
        agent_kwargs={
            "suffix": (
                "You are an expert Vehicle Data Analyst. "
                "Your job is to provide concise summaries and accurate insights based on the provided CSV data. "
                "Always generate and run the necessary Python code using pandas to answer the user's questions. "
                "Be direct and precise in your final answer."
            )
        }
    )
    return agent_executor

# --- 3. FILE UPLOAD SECTION ---
uploaded_file = st.sidebar.file_uploader("Upload your Vehicle Data CSV", type="csv")
df = None

if uploaded_file is not None:
    # Read the uploaded CSV file into a Pandas DataFrame
    data = uploaded_file.getvalue().decode("utf-8")
    df = pd.read_csv(StringIO(data))
    
    st.sidebar.success(f"File uploaded successfully: {uploaded_file.name}")
    st.sidebar.subheader("Data Preview (First 5 rows)")
    st.sidebar.dataframe(df.head(), use_container_width=True) 

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if df is not None:
    # Create or retrieve the agent
    agent = create_agent(df)

    if agent:
        # Accept user input
        if prompt := st.chat_input("Ask about your data (e.g., 'What is the average MPG?')"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        # Run the agent
                        response = agent.invoke({"input": prompt})
                        agent_response = response['output']
                    except Exception as e:
                        # Provide user-friendly error feedback
                        print(f"Agent Execution Error: {e}") 
                        agent_response = "I encountered an error during analysis. Please check your question or data for issues and try again."
                
                st.markdown(agent_response)
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": agent_response})

else:
    st.info("👆 Please upload a CSV file in the sidebar to begin interacting with the agent.")
