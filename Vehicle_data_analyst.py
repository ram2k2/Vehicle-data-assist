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
