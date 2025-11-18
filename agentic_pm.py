import streamlit as st
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. Setup Gemini API Key
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("GEMINI_API_KEY not found in Streamlit secrets.")
    gemini_api_key = None

# 2. Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=gemini_api_key
)

# 3. Agent Helper
def simple_agent(name: str, prompt_template: str) -> Runnable:
    def _agent(state: dict) -> dict:
        input_text = state.get("input", "")
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

# 4. Chief Agent
def chief_agent(state: dict) -> dict:
    input_text = state.get("input", "")
    csv_content = state.get("csv_content", "")

    decision_prompt = f"""
You are the Chief Agent. Based on the user's request and the CSV content, decide which of the following agents should be invoked:

- "summarizer"
- "insight_generator"
- "question_generator"
- "visualization_agent"

Only return the agent name(s) as a comma-separated list. Do not explain.

User input: {input_text}
CSV content (truncated): {csv_content[:1000]}
"""

    response = llm.invoke([HumanMessage(content=decision_prompt)])
    selected_agents = response.content.strip().split(",")
    next_agent = selected_agents[0].strip()

    return {
        **state,
        "next_agent": next_agent
    }

# 5. Define Agents
summarizer = simple_agent(
    "Summarizer Agent",
    """You are a summarization agent. Do not generate code or visualizations.
Summarize the cleaned vehicle data in 3–5 sentences. Include:
- 3 to 5 key metrics with units (e.g., total distance, fuel efficiency, battery SOH)
- 2 to 3 insights or trends (e.g., consistent speed, declining SOH, peak efficiency days)
- Use plain language and avoid technical jargon
- Format with short paragraphs or bullet points if needed

Add footer: 'Data extracted from {filename}'"""
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

# 6. Build LangGraph
graph_builder = StateGraph(dict)
graph_builder.set_entry_point("Chief Agent")

graph_builder.add_node("Chief Agent", chief_agent)
graph_builder.add_node("Summarizer", summarizer)
graph_builder.add_node("Insight Generator", insight_generator)
graph_builder.add_node("Question Generator", question_generator)
graph_builder.add_node("Visualization Agent", visualization_agent)

graph_builder.add_conditional_edges("Chief Agent", lambda state: state["next_agent"], {
    "summarizer": "Summarizer",
    "insight_generator": "Insight Generator",
    "question_generator": "Question Generator",
    "visualization_agent": "Visualization Agent"
})

graph_builder.add_edge("Summarizer", END)
graph_builder.add_edge("Insight Generator", END)
graph_builder.add_edge("Question Generator", END)
graph_builder.add_edge("Visualization Agent", END)

pm_graph = graph_builder.compile()

# 7. Run the Graph
def run_pm_agent(problem_statement: str, filename: str = "uploaded_data.csv", csv_content: str = "", follow_up: str = None):
    print("\nRunning Vehicle Data Analyst for:", problem_statement)
    state = {
        "input": problem_statement,
        "filename": filename,
        "csv_content": csv_content,
    }
    if follow_up:
        state["user_input"] = follow_up

    final_state = pm_graph.invoke(state)
    print("\nFinal Output:\n")
    print(final_state["output"])
    return final_state
