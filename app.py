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

if st.session_state.csv_content:
    st.subheader("📋 Summary & Insights")
    st.markdown(output["output"])

    with st.expander("🧠 Full Thought Process"):
        st.text(output["history"])

    st.markdown("### 🔍 Want to explore more?")
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Click here for more insights"):
            output = run_pm_agent("more insights", filename=st.session_state.filename, csv_content=st.session_state.csv_content)
            st.markdown(output["output"])

            import re
            st.session_state.suggested_questions = re.findall(r"- (.+)", output["output"])

            for q in st.session_state.suggested_questions:
                if st.button(q):
                    follow_up_output = run_pm_agent(q, filename=st.session_state.filename, csv_content=st.session_state.csv_content, follow_up=q)
                    st.markdown(follow_up_output["output"])

    with col2:
        user_question = st.text_input("Or enter your own question")
        if user_question:
            output = run_pm_agent(user_question, filename=st.session_state.filename, csv_content=st.session_state.csv_content, follow_up=user_question)
            st.markdown(output["output"])
        
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
