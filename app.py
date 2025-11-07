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
        "       - Use a bold title, e.g., **🔍 Vehicle Data Summary**\n"
        "       - Use numbered lists for each category.\n"
        "       - Include Start/End values for distance calculation.\n"
        "       - Add a contextual note (e.g., maintenance implications) where appropriate.\n"
        "4. **Visualization**: Generate relevant **charts and graphs** (using Matplotlib or similar if needed within the agent's Python execution) to visually represent key data points (e.g., speed distribution, distance over time).\n"
        "5. **Constraints**: Refrain from making assumptions or providing insights not supported by the data. Acknowledge any data limitations or gaps found.\n"
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
    return agent_executor

# --- 3. STREAMLIT APP UI ---
st.set_page_config(page_title="Vehicle Data Analyst", layout="wide")
st.title("🚗 Vehicle Data Analyst Agent")

# Initial Welcome Message
if not st.session_state.get("initial_greeting_sent", False):
    st.markdown("""
        Hi, welcome! 😊
        Please upload a CSV file for vehicle data analysis. 
        **Use the semicolon (`;`) delimiter for parsing the file.**
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
                    # Run the agent
                    response = agent.invoke({"input": prompt})
                    agent_response = response['output']
                except Exception as e:
                    # Log the detailed error for debugging purposes
                    print(f"Agent Execution Error: {e}")
                    agent_response = "An unexpected error occurred during analysis. Please try simplifying your request or checking the column names."
            
            st.markdown(agent_response)
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

else:
    # Use a simpler prompt for the user when waiting for a file
    pass
