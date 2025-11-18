# app.py
import streamlit as st
from agentic_pm import run_pm_agent

st.set_page_config(page_title="Vehicle Data Assist", layout="wide")

st.title("🧠 Vehicle Data Assist")
st.markdown("""
Upload a vehicle data CSV and this AI-powered system will automatically analyze it, summarize key metrics, and suggest follow-up questions.
""")

# File upload
filename = ""
csv_content = ""
uploaded_file = st.file_uploader("📁 Upload vehicle data CSV", type=["csv"])
if uploaded_file:
    csv_content = uploaded_file.read().decode("utf-8")
    filename = uploaded_file.name

    # Auto-run agent when file is uploaded
    with st.spinner("Analyzing vehicle data..."):
        output = run_pm_agent("Vehicle data analysis", filename=filename, csv_content=csv_content)
        st.success("Summary generated!")

        st.subheader("📋 Summary & Insights")
        st.markdown(output["output"])

        with st.expander("🧠 Full Thought Process"):
            st.text(output["history"])

# Follow-up input
follow_up = st.chat_input("Ask a follow-up question")
if follow_up and csv_content:
    output = run_pm_agent("Vehicle data analysis", filename=filename, csv_content=csv_content, follow_up=follow_up)
    st.markdown("### 🤖 Follow-Up Response")
    st.markdown(output["output"])
