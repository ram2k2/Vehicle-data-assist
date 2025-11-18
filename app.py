import streamlit as st
import json
import requests
import time

# --- Configuration ---
# API Key is left empty as the Canvas environment injects it at runtime.
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --- Agent Core Function ---

def run_pm_agent_gemini(problem_statement: str) -> dict:
    """
    Calls the Gemini API to analyze a problem statement, acting as a Product Manager.
    Uses Google Search grounding for real-time market analysis.
    """
    
    # 1. Define the PM Persona and Task
    system_instruction = {
        "parts": [{
            "text": "You are a seasoned, strategic Product Manager. Your task is to analyze the user's problem statement. Provide a structured response covering: 1. **Problem Description** (who experiences it), 2. **Market Context** (current trends and size), 3. **Potential Solution** (a high-level feature concept), and 4. **Key Metrics** (how success will be measured). Write the entire output using Markdown headings."
        }]
    }
    
    # 2. Define the User Query
    user_query = f"Analyze the following product problem statement: {problem_statement}"

    # 3. Construct the API Payload
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": system_instruction,
        "tools": [{"google_search": {}}], # Enable search grounding
    }

    # 4. Make the API Call with Retry Logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Extract the generated text
            output_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Error: Could not retrieve response text.')
            
            # Mock structure to match the user's original reference (history and output keys)
            return {
                "output": output_text,
                "history": f"Gemini ran the Product Manager analysis on: '{problem_statement}'. Search grounding was used for market research."
            }

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                # print(f"API call failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {
                    "output": f"Error: Failed to connect to the analysis engine after {max_retries} attempts. Details: {e}",
                    "history": f"API call failed: {e}"
                }
    
    return {"output": "An unexpected error occurred.", "history": "Unknown error in API call logic."}


# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Agentic Product Manager", layout="wide")

    st.title("🧠 Agentic Product Manager (Gemini Powered)")
    st.markdown("""
    Enter a product problem statement, and this AI-powered system will act like a seasoned Product Manager — analyzing the market, identifying solutions, and defining success metrics.
    """)

    problem_input = st.text_area(
        "📝 Product Problem Statement", 
        placeholder="e.g. How can we reduce churn for users who complete the first five levels of our mobile game?"
    )

    if st.button("🚀 Run Agentic PM"):
        if not problem_input.strip():
            st.warning("Please enter a valid problem statement.")
        else:
            # 1. Run the agent and show a spinner
            with st.spinner("Thinking like a PM and researching the market..."):
                output = run_pm_agent_gemini(problem_input)
            
            st.success("Agentic PM analysis completed!")

            # 2. Display the 'Full PM Thought Process' (mocked history)
            with st.expander("🧠 Full PM Thought Process"):
                st.text(output["history"])
            
            # 3. Display the final analysis output
            st.subheader("📋 Final Analysis Output")
            st.markdown(output["output"])

if __name__ == "__main__":
    main()
