import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

# --- 1. CONFIGURATION AND SECRETS ---

# Set up the Streamlit page title and layout
st.set_page_config(
    page_title="AI Vehicle Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Access the API key securely from Streamlit's secrets management
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
    MODEL = 'gemini-2.5-flash'  # Use a cost-effective model for the MVP
except KeyError:
    st.error("Gemini API Key not found. Please set 'GEMINI_API_KEY' in your Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop()

# --- 2. DATA ANALYSIS CORE (Caching for performance) ---

# Use st.cache_data to prevent re-running the heavy analysis every time the app updates
@st.cache_data
def perform_initial_analysis(df):
    """
    Cleans the DataFrame and calculates the four core vehicle metrics.
    """
    required_cols = {
        "Total distance (km)": "distance",
        "Fuel efficiency": "fuel_eff",
        "High voltage battery State of Health (SOH).": "soh",
        "Current vehicle speed.": "speed"
    }
    
    # 1. Cleaning: Replace invalid entries and convert to numeric
    for col in required_cols.keys():
        if col in df.columns:
            # Replace invalid entries ("NV", "NA", empty) with NaN
            df[col] = df[col].replace(['NV', 'NA', '', ' '], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            st.warning(f"Required column '{col}' not found. Skipping calculations for this metric.")
            return None, "Error: Missing required column(s)."
    
    # Drop rows with NaN *only* for calculation purposes, using the cleaned series
    df_clean = df.dropna(subset=required_cols.keys())

    if df_clean.empty:
        return None, "Analysis failed: Data contains no valid entries after cleaning."

    # 2. Calculation
    
    # Total Distance = last - first value of "Total distance (km)"
    total_distance_driven = df_clean["Total distance (km)"].iloc[-1] - df_clean["Total distance (km)"].iloc[0]
    
    # Average Fuel Efficiency = mean of "Fuel efficiency"
    avg_fuel_efficiency = df_clean["Fuel efficiency"].mean()
    
    # Latest Battery SOH = last value of "High voltage battery State of Health (SOH)."
    latest_soh = df_clean["High voltage battery State of Health (SOH)."].iloc[-1]
    
    # Average Vehicle Speed = mean of "Current vehicle speed."
    avg_vehicle_speed = df_clean["Current vehicle speed."].mean()

    # 3. Create a structured text summary for the LLM prompt (and display)
    
    # Using the LaTeX math environment for clarity and Markdown for bolding
    format_value = lambda val, unit: f"**{val:,.2f}** {unit}" if pd.notna(val) else f"**---** {unit}"
    
    metrics_summary = (
        f"| Metric | Calculated Value |\n"
        f"| :--- | :--- |\n"
        f"| Total Distance Driven | {format_value(total_distance_driven, 'km')} |\n"
        f"| Average Fuel Efficiency | {format_value(avg_fuel_efficiency, 'km/L or equivalent')} |\n"
        f"| Latest Battery SOH | {format_value(latest_soh, '\\%')} |\n"
        f"| Average Vehicle Speed | {format_value(avg_vehicle_speed, 'km/h')} |\n"
    )
    
    # Return the data for future use and the LLM input string
    return df_clean, metrics_summary

# --- 3. GEMINI AGENT LOGIC (Initial Summary and Suggested Questions) ---

@st.cache_data(show_spinner="Analyzing data and generating insights with Gemini...")
def get_gemini_analysis(metrics_summary):
    """
    Sends the calculated metrics to Gemini and retrieves the conversational summary.
    """
    # System instruction sets the persona and goal
    system_instruction = (
        "You are an expert Vehicle Data Analyst. Your goal is to provide a clear, "
        "conversational summary and proactively suggest 3-5 critical and interesting "
        "follow-up questions the user should ask to explore anomalies or patterns in their data. "
        "Do not output code. Present the analysis in Markdown."
    )
    
    # User prompt containing the structured data
    prompt = (
        "Analyze the following core vehicle metrics. \n"
        "First, generate a concise, user-friendly summary of this data. \n"
        "Second, provide a bulleted list of 3-5 suggested questions based on these specific values. \n"
        "Do not repeat the table in your final response.\n\n"
        "## Vehicle Metrics\n"
        f"{metrics_summary}"
    )
    
    # Call the Gemini API
    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2  # Lower temperature for factual analysis
        )
    )
    
    return response.text

# --- 4. STREAMLIT UI LAYOUT ---

st.title("🚗 AI Vehicle Data Analyst Agent")
st.markdown("Upload your vehicle data extract (CSV) to get an automated analysis, insights, and suggested questions from the AI Agent.")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV Data Extract", type=["csv"])

if uploaded_file is not None:
    # Read and analyze data when a file is uploaded
    df = pd.read_csv(uploaded_file)
    
    # Perform the data cleaning and calculation
    df_clean, metrics_summary = perform_initial_analysis(df)
    
    if df_clean is not None:
        
        st.header("🔍 Initial Data Snapshot")
        st.markdown(metrics_summary) # Display the calculated table
        
        st.divider()

        # Get the conversational summary from Gemini
        try:
            gemini_output = get_gemini_analysis(metrics_summary)
            
            st.header("🤖 Agent's Summary and Suggested Questions")
            st.markdown(gemini_output)
            
            # Initialize a chat interface for the next conversational step
            st.subheader("Start the Conversation:")
            st.chat_input(
                placeholder="Ask one of the suggested questions or your own query...", 
                key="user_chat_input"
            )
            
        except Exception as e:
            st.error(f"Error communicating with the Gemini API: {e}. Check your usage limits.")
    else:
        st.error(metrics_summary) # Display the error message from analysis