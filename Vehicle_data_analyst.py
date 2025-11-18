import pandas as pd
import io
from langchain.tools import tool
# ... other imports (os, streamlit, genai) ...

# --- Data Handling Functions ---

def clean_and_calculate_summary(data_string: str, filename: str) -> dict:
    # Use io.StringIO to treat the string as a file
    data_io = io.StringIO(data_string)
    # Manual parsing: read CSV, explicitly using ';' as delimiter
    df = pd.read_csv(data_io, sep=';', skipinitialspace=True)
    
    # Define columns to process
    cols_to_clean = ["Total distance (km)", "Fuel efficiency", 
                     "High voltage battery State of Health (SOH)", "Current vehicle speed"]
    
    clean_data = {}
    
    for col in cols_to_clean:
        # Convert to numeric, errors='coerce' turns "NV", "NA", etc. into NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows where the value is NaN for this specific calculation
        clean_values = df[col].dropna()
        clean_data[col] = clean_values

    # 1. Total Distance
    total_distance = (clean_data["Total distance (km)"].iloc[-1] - 
                      clean_data["Total distance (km)"].iloc[0]) if not clean_data["Total distance (km)"].empty else 0

    # 2. Average Fuel Efficiency
    avg_fuel_efficiency = clean_data["Fuel efficiency"].mean()
    
    # 3. Latest Battery SOH
    latest_soh = clean_data["High voltage battery State of Health (SOH)"].iloc[-1] if not clean_data["High voltage battery State of Health (SOH)"].empty else "N/A"
    
    # 4. Average Vehicle Speed
    avg_speed = clean_data["Current vehicle speed"].mean()

    # --- Structured Output ---
    summary = f"""
**Vehicle Data Summary (from {filename}):**
* **Total Trip Distance:** **{total_distance:.2f} km**
* **Average Fuel Efficiency:** **{avg_fuel_efficiency:.2f} L/100km**
* **Latest Battery SOH:** **{latest_soh}%**
* **Average Vehicle Speed:** **{avg_speed:.2f} km/h**
"""
    
    return {"summary_text": summary, "dataframe": df}

# --- LangChain Tool Definition ---

@tool
def data_processor_tool(data_extract: str, filename: str) -> str:
    """
    Analyzes a raw CSV string of vehicle data.
    It cleans the data (removes 'NV', 'NA', empty strings), calculates total distance, 
    average fuel efficiency, latest battery SOH, and average speed. 
    Returns the structured text summary for the LLM to use.
    """
    result = clean_and_calculate_summary(data_extract, filename)
    # Store the dataframe in session state for the visualization tool
    st.session_state['processed_df'] = result['dataframe']
    return result['summary_text']

import matplotlib.pyplot as plt

@tool
def visualization_tool(column_name: str, filename: str) -> str:
    """
    Creates a time-series plot (or distribution) for a specified column from the data.
    It saves the chart as an image and returns a path/placeholder for the Streamlit app to display.
    Only call this tool when the user asks for a chart or graph.
    """
    if 'processed_df' not in st.session_state:
        return "Error: Data has not been processed yet. Run the summary first."

    df = st.session_state['processed_df']
    
    # Check if column exists and is numeric
    if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
        return f"Error: Cannot visualize '{column_name}'. It may not exist or is not numeric."
    
    # Simple Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    df[column_name].plot(ax=ax, title=f'{column_name} Over Time')
    plt.xlabel('Time Step')
    plt.ylabel(column_name)
    plt.tight_layout()
    
    # Save the figure to a buffer and return a placeholder name
    st.session_state['chart_fig'] = fig
    return f"Chart for '{column_name}' has been prepared. Please instruct the user to refresh the Streamlit display."

# --- Agent Initialization (`vehicle_agent.py` continued) ---

# Initialize Gemini (Ensure API key is set in Streamlit Secrets or environment)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # Use a low temp for stable analysis

# Combine the tools
tools = [data_processor_tool, visualization_tool]

# The main Agent Executor
prompt = """
You are an expert Vehicle Data Analyst. Your goal is to analyze vehicle data and provide clear, conversational, and useful insights.

1. **ALWAYS** start by using the `data_processor_tool` to get the summary statistics and clean the data.
2. After presenting the summary, **ALWAYS** suggest 2-3 new, insightful questions the user could ask based on the data provided.
3. If the user asks for a chart or visualization, use the `visualization_tool`.
4. **ALWAYS** add the footer: "Data extracted from (filename)" to your final response.
"""

# Initialize the Agent
vehicle_data_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

def run_agent(problem_statement: str, data_string: str, filename: str) -> str:
    # The agent needs the data and filename to be passed to the tool
    # We embed the data and filename into the prompt for the agent to use the tool
    full_prompt = (
        f"{prompt}\n\n"
        f"DATA_EXTRACT_CSV_STRING:\n---\n{data_string}\n---\n"
        f"FILENAME: {filename}\n\n"
        f"USER_REQUEST: {problem_statement}"
    )
    
    response = vehicle_data_agent.run(full_prompt)
    
    return response
