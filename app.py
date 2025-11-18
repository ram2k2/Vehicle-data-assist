# app.py
import streamlit as st
from agentic_pm import run_pm_agent

st.set_page_config(page_title="Agentic Product Manager", layout="wide")

st.title("🧠 Agentic Product Manager")
st.markdown("""
Enter a product problem statement, and this AI-powered system will act like a seasoned Product Manager — breaking down the problem, analyzing the market, identifying competitors, writing PRDs, and more!
""")

problem_input = st.text_area("📝 Problem Statement", placeholder="e.g. Improve onboarding for first-time EV users")

if st.button("🚀 Run Agentic PM"):
    if not problem_input.strip():
        st.warning("Please enter a valid problem statement.")
    else:
        with st.spinner("Thinking like a PM..."):
            
            output = run_pm_agent(problem_input)
            st.success("Agentic PM completed!")

            # Expandable section for entire trace
            with st.expander("🧠 Full PM Thought Process"):
                st.text(output["history"])
            
            st.subheader("📋 Final Output")
            st.markdown(output["output"])
