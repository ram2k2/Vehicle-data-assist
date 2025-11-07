import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from io import StringIO
from dotenv import load_dotenv

# Load environment variables (for local testing)
load_dotenv() 

# --- 1. CONFIGURATION ---
MODEL_NAME = "gemini-2.5-flash"

# --- 2. AGENT CREATION FUNCTION ---
# The LLM and agent are cached to avoid expensive re-initialization.
@st.cache_resource
def create_agent(df: pd.DataFrame):
    """Initializes and returns the LangChain Pandas DataFrame Agent."""
    
    # Check for API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # In a Streamlit Cloud context, rely on the environment's handling of the key.
        pass

    # LLM Initialization
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, 
        # Pass api_key if it was successfully retrieved, otherwise rely on the environment
        api_key=api_key if api_key else None
    )
    
    # Custom, highly detailed System Prompt (Suffix) to enforce required behavior
    SYSTEM_PROMPT_SUFFIX = (
    "You are an expert **Vehicle Data Analyst** working with data obtained from Data Act access requests. "
    "Your ultimate goal is to provide **meaningful, useful, and actionable insights** that help the user understand their vehicle's health, efficiency, and usage. "
    "Your responses must be **formal, direct, and authoritative**. "
    "--- Mandatory Protocol ---\n"
    "1. **Execution**: You **MUST** generate and run the necessary Python code using pandas to answer questions.\n"
    "2. **Initial Check**: Immediately state the exact row and column count of the loaded DataFrame.\n"
    "3. **Summary Command**: When the user requests a 'summary' or 'comprehensive breakdown', follow these steps precisely:\n"
    "    a. **Data Cleaning**: Handle potential inconsistencies. Locate the columns: `Total distance (km)`, `Fuel efficiency`, `High voltage battery State of Health (SOH).`, and `Current vehicle speed.`. Convert these columns to numeric format (errors='coerce') and drop any rows with missing or invalid data (NV) for these target columns.\n"
    "    b. **Calculations**: Perform the following calculations:\n"
    "        - **Total Distance Traveled**: Calculate the difference between the final and initial values in the `Total distance (km)` column.\n"
    "        - **Average Fuel Efficiency**: Compute the mean of all valid values in the `Fuel efficiency` column.\n"
    "        - **Latest Battery SOH**: Use the most recent value (last row) from the `High voltage battery State of Health (SOH).` column.\n"
    "        - **Average Vehicle Speed**: Calculate the mean of the `Current vehicle speed.` column.\n"
    "    c. **Formatted Insightful Response**: After calculations, compile the results into a clear, professional summary using the exact structure below. **DO NOT use df.describe() or output generic summary statistics.** The final Python code output must be a print statement that generates the complete, formatted Markdown response.\n"
    "Example Output:\n"
    "**🔍 Vehicle Data Summary**\n"
    "1. **Total Distance Traveled**: 1000 km (from 0 km to 1000 km)\n"
    "2. **Average Fuel Efficiency**: 15.3 km/l\n"
    "3. **Latest Battery SOH**: 97%\n"
    "4. **Average Vehicle Speed**: 50 km/h\n"
    "Note: The vehicle's battery health is good, and the fuel efficiency is average.\n"
    "---\n"
    "Please provide the Python code to perform these calculations and generate the summary."
)

    # The LangChain Pandas DataFrame Agent setup
agent_executor = create_pandas_dataframe_agent(
    llm=llm,
    df=df,  # Pass the DataFrame directly
    verbose=False,
    agent_type="openai-tools",
    # CRITICAL FIX: Allows Python code execution in sandboxed environment
    allow_dangerous_code=True, 
    agent_kwargs={
        "suffix": SYSTEM_PROMPT_SUFFIX
    }
)

# --- 3. STREAMLIT APP UI ---
st.set_page_config(page_title="Vehicle Data Analyst", layout="wide")
st.title("🚗 Vehicle Data Analyst Agent")

# Initial Welcome Message
if not st.session_state.get("initial_greeting_sent", False):
    st.markdown("""
        Hi, welcome! 😊
        Please upload a CSV file for vehicle data analysis. 
    """)
    st.session_state["initial_greeting_sent"] = True


# --- File Upload Section ---
uploaded_file = st.sidebar.file_uploader("Upload your Vehicle Data CSV", type="csv")
df = None

if uploaded_file is not None:
    # Read the uploaded CSV file into a Pandas DataFrame using SEMICOLON (;) delimiter
    data = uploaded_file.getvalue().decode("utf-8")
    # CRITICAL: Added sep=';' to enforce semicolon delimiter
    try:
        df = pd.read_csv(StringIO(data), sep=';')
        st.sidebar.success(f"File uploaded successfully: {uploaded_file.name}")
        st.sidebar.dataframe(df.head(), use_container_width=True) # Show a preview in the sidebar
    except Exception as e:
        st.sidebar.error(f"Error reading CSV with ';' delimiter. Check your file format. Error: {e}")
        df = None


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
    if prompt := st.chat_input("Ask about your data (e.g., 'Give me a comprehensive summary')"):
        
        if agent is None: # Handle case where API key is missing after file upload
            st.warning("Cannot run analysis. Please ensure your GEMINI_API_KEY is configured.")
            st.stop()
            
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    response = agent.invoke({"input": prompt})
                    agent_response = response['output']
                except Exception as e:
                    agent_response = f"An error occurred: {str(e)}"
            
            st.markdown(agent_response)
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

else:
    # Use a simpler prompt for the user when waiting for a file
    pass
