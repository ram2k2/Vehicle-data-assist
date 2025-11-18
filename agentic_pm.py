import streamlit as st
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. Setup Gemini API Key
try:
    # Safely retrieve the Gemini API key from Streamlit secrets
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Handle the missing key gracefully
    st.error("GEMINI_API_KEY not found in Streamlit secrets.")
    gemini_api_key = None
    
# 2. Initialize LLM (Using Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    google_api_key=gemini_api_key
)

# 3. Agent Helper Functions

def simple_agent(name: str, prompt_template: str) -> Runnable:
    def _agent(state: dict) -> dict:
        input_text = state["input"]
        history = state.get("history", "")
        csv_content = state.get("csv_content", "")
        filename = state.get("filename", "uploaded_data.csv")

        full_prompt = f"{prompt_template}\n\nCSV Content:\n{csv_content}\n\nProblem Statement: {input_text}\n\n{history}"
        full_prompt = full_prompt.replace("{filename}", filename)

        response = llm.invoke([HumanMessage(content=full_prompt)])
        return {
            "input": input_text,
            "history": history + f"\n\n[{name}]\n{response.content}",
            "output": response.content,
        }
    return _agent

# 4. Define All Agents

data_preprocessor = simple_agent(
    "Data Preprocessor Agent",
    """Parse the uploaded CSV using ';' as delimiter. Clean invalid entries ("NV", "NA", empty).
Extract and calculate:
- Total Distance = last - first value of "Total distance (km)"
- Average Fuel Efficiency = mean of "Fuel efficiency"
- Latest Battery SOH = last value of "High voltage battery State of Health (SOH)."
- Average Vehicle Speed = mean of "Current vehicle speed."

Return a structured summary with bold formatting and units. Add footer: 'Data extracted from {filename}'."""
)

summarizer = simple_agent(
    "Summarizer Agent",
    "Summarize the cleaned vehicle data in a structured format. Highlight key metrics and trends. Add footer: 'Data extracted from {filename}'"
)

insight_generator = simple_agent(
    "Insight Generator Agent",
    "Analyze the summary and generate interesting insights, patterns, or anomalies. Add footer: 'Data extracted from {filename}'"
)

question_generator = simple_agent(
    "Question Generator Agent",
    """Based on the insights, suggest 3–5 follow-up questions in a conversational tone, e.g.:
- 'Would you like to check the trend of oil temperature over the last 7 days?'
- 'Should we explore how fuel efficiency varied across different driving speeds?'
Add footer: 'Data extracted from {filename}'"""
)

visualization_agent = simple_agent(
    "Visualization Agent",
    "Generate visualizations for key metrics like distance, fuel efficiency, battery SOH, and speed. Add footer: 'Data extracted from {filename}'"
)



# 5. Build LangGraph
graph_builder = StateGraph(dict)
graph_builder.set_entry_point("Data Preprocessor")

graph_builder.add_node("Data Preprocessor", data_preprocessor)
graph_builder.add_node("Summarizer", summarizer)
graph_builder.add_node("Insight Generator", insight_generator)
graph_builder.add_node("Question Generator", question_generator)
graph_builder.add_node("Visualization Agent", visualization_agent)

graph_builder.add_edge("Data Preprocessor", "Summarizer")
graph_builder.add_edge("Summarizer", "Insight Generator")
graph_builder.add_edge("Insight Generator", "Question Generator")
graph_builder.add_edge("Question Generator", "Visualization Agent")
graph_builder.add_edge("Visualization Agent", END)

pm_graph = graph_builder.compile()

# 6. Run the Graph

def run_pm_agent(problem_statement: str, filename: str = "uploaded_data.csv", csv_content: str = "", follow_up: str = None):
    print("\n Running Vehicle Data Analyst for:", problem_statement)
    state = {
        "input": problem_statement,
        "filename": filename,
        "csv_content": csv_content,
    }
    if follow_up:
        state["user_input"] = follow_up

    final_state = pm_graph.invoke(state)
    print("\n Final Output:\n")
    print(final_state["output"])
    return final_state
