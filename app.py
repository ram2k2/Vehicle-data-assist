import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from io import StringIO
from dotenv import load_dotenv

# Load environment variables (useful for local testing, Streamlit handles keys in production)
load_dotenv() 

# --- 1. CONFIGURATION ---
MODEL_NAME = "gemini-2.5-flash"

# --- 2. AGENT CREATION FUNCTION ---
# This function is cached to avoid recreating the agent/LLM on every user interaction
@st.cache_resource
def create_agent(df: pd.DataFrame):
    """Initializes and returns the LangChain CSV Agent."""
    # Ensure GEMINI_API_KEY is available in the environment (set in Streamlit Secrets)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY environment variable not set. Cannot run the agent.")
        return None

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=api_key)
    
    # The LangChain CSV Agent setup
    agent_executor = create_csv_agent(
        llm=llm,
        path=df,  # Pass the DataFrame directly to the agent
        verbose=False,
        agent_type="openai-tools",
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

# --- 3. STREAMLIT APP UI ---
st.set_page_config(page_title="Free Vehicle Data Agent", layout="wide")
st.title("🚗 Free Vehicle Data Insights Agent")
st.write("Upload your CSV file to get instant summaries and insights using the Gemini AI model.")

# --- File Upload Section ---
uploaded_file = st.sidebar.file_uploader("Upload your Vehicle Data CSV", type="csv")
df = None

if uploaded_file is not None:
    # Read the uploaded CSV file into a Pandas DataFrame
    data = uploaded_file.getvalue().decode("utf-8")
    df = pd.read_csv(StringIO(data))
    
    st.sidebar.success(f"File uploaded successfully: {uploaded_file.name}")
    st.sidebar.dataframe(df.head(), use_container_width=True) # Show a preview in the sidebar

# --- Chat Interface Section ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if df is not None:
    # Create or retrieve the agent
    agent = create_agent(df)

    # Accept user input
    if prompt := st.chat_input("Ask about your data (e.g., 'What is the average speed?')"):
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
                    agent_response = f"An error occurred during analysis: {e}. Please check your data format or try a simpler question."
            
            st.markdown(agent_response)
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

else:
    st.info("👆 Please upload a CSV file in the sidebar to begin interacting with the agent.")
