# app.py
import streamlit as st
from dev_agentic_pm import run_pm_agent

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
st.set_page_config(page_title="Vehicle Data Assist", layout="wide")

st.title("🧠 Vehicle Data Assist")
st.markdown("""
Upload a vehicle data CSV and this AI-powered system will automatically analyze it, summarize key metrics, and suggest follow-up questions.
""")

# File upload
if "filename" not in st.session_state:
    st.session_state.filename = ""
if "csv_content" not in st.session_state:
    st.session_state.csv_content = ""
    
uploaded_file = st.file_uploader("📁 Upload the csv file", type=["csv"])
if uploaded_file:
    st.session_state.csv_content = uploaded_file.read().decode("utf-8")
    st.session_state.filename = uploaded_file.name

    # Auto-run agent when file is uploaded
    with st.spinner("Analyzing vehicle data..."):
        output = run_pm_agent("Vehicle data analysis", filename=st.session_state.filename, csv_content=st.session_state.csv_content)
        st.success("Summary generated!")

        st.subheader("📋 Summary & Insights")
        st.markdown(output["output"])

        with st.expander("🧠 Full Thought Process"):
            st.text(output["history"])

# Follow-up input
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(chat["query"])
    with st.chat_message("assistant"):
        st.markdown(chat["response"])
        
follow_up = st.chat_input("Ask a follow-up question")
if follow_up and st.session_state.csv_content:
    output = run_pm_agent("Vehicle data analysis", filename=st.session_state.filename, csv_content=st.session_state.csv_content, follow_up=follow_up)
    response_text = output["output"]

    # Save to history
    st.session_state.chat_history.append({
        "query": follow_up,
        "response": response_text
    })

    # Display latest response
    with st.chat_message("user"):
        st.markdown(follow_up)
    with st.chat_message("assistant"):
        st.markdown(response_text)

if st.sidebar.button("🔄 Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

if st.session_state.filename:
    st.sidebar.caption(f"📄 File: `{st.session_state.filename}`")
