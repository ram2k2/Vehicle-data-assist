import streamlit as st
from agent import handle_query, analyze_csv

st.set_page_config(page_title="Vehicle Data Assist - V1", layout="wide")

st.title("🚗 Vehicle Data Assist - Version 1")
st.write("Upload a semicolon-delimited CSV and get a quick summary. Then ask questions!")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")
    with open("uploaded.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("Generating summary...")
    summary = analyze_csv("uploaded.csv")
    st.info(summary)

# Chat interface
st.subheader("Ask a question:")
user_query = st.text_input("Enter your query")

if st.button("Submit"):
    if user_query.strip():
        response = handle_query(user_query)
        st.write("### Response:")
        st.write(response)
