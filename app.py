import streamlit as st
from agentic_pm import run_pm_agent

st.set_page_config(page_title="Vehicle Data Assist", layout="wide")
st.title("🧠 Vehicle Data Assist")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# File upload
filename = ""
csv_content = ""
uploaded_file = st.file_uploader("📁 Upload vehicle data CSV", type=["csv"])

if uploaded_file:
    csv_content = uploaded_file.read().decode("utf-8")
    filename = uploaded_file.name

    with st.spinner("Analyzing vehicle data..."):
        output = run_pm_agent("Vehicle data analysis", filename=filename, csv_content=csv_content)
        st.session_state["messages"].append({"role": "assistant", "content": output["output"]})

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input for follow-up
follow_up = st.chat_input("Ask a follow-up question")
if follow_up and csv_content:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": follow_up})

    # Get agent response
    output = run_pm_agent("Vehicle data analysis", filename=filename, csv_content=csv_content, follow_up=follow_up)
    st.session_state["messages"].append({"role": "assistant", "content": output["output"]})

    # Display immediately
    with st.chat_message("assistant"):
        st.markdown(output["output"])
