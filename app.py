# app.py
import streamlit as st
from agentic_pm import run_pm_agent

st.set_page_config(page_title="Vehicle Data Assist", layout="wide")

st.title("🧠 Vehicle Data Assist")
st.markdown("""
Upload a vehicle data CSV and enter a problem statement. This AI-powered system will analyze the data, generate insights, and suggest follow-up questions!
""")

# File upload
filename = ""
csv_content = ""
uploaded_file = st.file_uploader("Upload vehicle data CSV", type=["csv"])
if uploaded_file:
    csv_content = uploaded_file.read().decode("utf-8")
    filename = uploaded_file.name

# Problem input
problem_input = st.text_area("📝 Problem Statement", placeholder="e.g. Analyze driving behavior over the past month")

# Auto-run when both file and input are present
if problem_input.strip() and csv_content:
    with st.spinner("Analyzing vehicle data..."):
        output = run_pm_agent(problem_input, filename=filename, csv_content=csv_content)
        st.success("Analysis complete!")

        st.subheader("📋 Final Output")
        st.markdown(output["output"])

        with st.expander("🧠 Full Thought Process"):
            st.text(output["history"])

# Follow-up input
follow_up = st.chat_input("Ask a follow-up question")
if follow_up and problem_input and csv_content:
    output = run_pm_agent(problem_input, filename=filename, csv_content=csv_content, follow_up=follow_up)
    st.markdown("### 🤖 Follow-Up Response")
    st.markdown(output["output"])
