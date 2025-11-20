import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from agent import create_agent, create_workflow

os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
# --- Streamlit UI ---
st.set_page_config(page_title="Vehicle Data Assist - Grand Version", layout="wide")
st.title("🚗 Vehicle Data Assist - Grand Version")

# Upload CSV
uploaded_file = st.file_uploader("Upload your vehicle data (CSV)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, delimiter=";")
    except Exception:
        st.error("Error parsing CSV. Trying default delimiter.")
        df = pd.read_csv(uploaded_file)
else:
    st.warning("No file uploaded. Using Test.csv fallback.")
    df = pd.read_csv("Test.csv", delimiter=";")

st.write("### Preview of Data", df.head())

# --- Visualization Section ---
st.subheader("Visualizations")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

if len(numeric_cols) > 0:
    # Histogram
    fig_hist = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Trend line (if date column exists)
    if 'Date' in df.columns or 'date' in df.columns:
        date_col = 'Date' if 'Date' in df.columns else 'date'
        fig_trend = px.line(df, x=date_col, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]} over time")
        st.plotly_chart(fig_trend, use_container_width=True)

    # Correlation heatmap
    corr = df[numeric_cols].corr()
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='Viridis'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # KPI cards
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average " + numeric_cols[0], round(df[numeric_cols[0]].mean(), 2))
    col2.metric("Max " + numeric_cols[0], round(df[numeric_cols[0]].max(), 2))
    col3.metric("Min " + numeric_cols[0], round(df[numeric_cols[0]].min(), 2))
else:
    st.info("No numeric columns detected for visualization.")

# --- LangChain Agent Setup ---
agent = create_agent(df)

# --- LangGraph Workflow ---
graph = create_workflow(df)
workflow_output = graph.run()

st.write("### Workflow Output", workflow_output)

# --- Chat Interface ---
st.subheader("Ask me anything about your data:")
user_query = st.text_input("Your question:")
if user_query:
    response = agent.run(user_query)
    st.write("**Response:**", response)

# --- Proactive Insights ---
st.subheader("Suggested Questions")
for q in workflow_output["suggest_questions"]:
    st.button(q)
