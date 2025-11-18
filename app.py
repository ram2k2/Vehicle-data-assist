# app.py
import streamlit as st
from vehicle_agent import run_agent

st.set_page_config(page_title="Vehicle Data Analyst", layout="wide")

st.title("🚗 AI Vehicle Data Analyst")
st.markdown("Upload a vehicle data CSV (semicolon-delimited) and ask for insights.")

# File Uploader
uploaded_file = st.file_uploader("Upload CSV Data (Delimiter: ';')", type=["csv"])

if uploaded_file:
    # Read file content
    file_contents = uploaded_file.getvalue().decode("utf-8")
    filename = uploaded_file.name

    # Text Input for the user's question
    user_query = st.text_input(
        "📝 Your Question for the Data:",
        placeholder="e.g. Give me the full summary and suggest some insights."
    )
    
    if st.button("Analyze Data"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            # Clear previous chart
            if 'chart_fig' in st.session_state:
                del st.session_state['chart_fig']
                
            with st.spinner("Analyzing data and generating insights..."):
                # Run the agent
                response = run_agent(user_query, file_contents, filename)
                
                st.success("Analysis Complete!")
                st.subheader("💡 Agent Response")
                st.markdown(response)

            # Check if the visualization tool prepared a chart
            if 'chart_fig' in st.session_state:
                st.subheader("📊 Visualization")
                st.pyplot(st.session_state['chart_fig'])
