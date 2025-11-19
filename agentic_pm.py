# agentic_pm.py
import streamlit as st
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load API key
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize LLM
llm = ChatGroq(model="Llama3-70b-8192", groq_api_key=groq_api_key)

# -------------------------------
# 1. Simple Agent Factory
# -------------------------------
def simple_agent(name: str, prompt_template: str):
    def _agent(state: dict):
        input_text = state["input"]
        history = state.get("history", "")
        full_prompt = f"{prompt_template}\n\nProblem Statement: {input_text}\n\n{history}"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        return {
            "input": input_text,
            "history": history + f"\n\n[{name}]\n{response.content}",
            "output": response.content,
        }
    return _agent

# -------------------------------
# 2. Define Nodes
# -------------------------------
problem_framer = simple_agent(
    "Problem Framer Agent",
    "Break the problem down into user impact, business goals, and tech implications. Format as bullet points."
)

market_analyst = simple_agent(
    "Market Analyst Agent",
    "Search the web to identify relevant market trends and summarize insights."
)

competitor_scout = simple_agent(
    "Competitor Scout Agent",
    "Find top competitors and summarize their features and pricing."
)

solution_designer = simple_agent(
    "Solution Designer Agent",
    "Suggest 2 product solutions with clear MVP scope."
)

prioritization_planner = simple_agent(
    "Prioritization Planner Agent",
    "Prioritize features using RICE scoring and propose a roadmap."
)

prd_writer = simple_agent(
    "PRD Writer Agent",
    "Write a concise PRD with Overview, Objectives, Features, and KPIs."
)

# -------------------------------
# 3. Build LangGraph
# -------------------------------
graph_builder = StateGraph(dict)
graph_builder.set_entry_point("Problem Framer")

graph_builder.add_node("Problem Framer", problem_framer)
graph_builder.add_node("Market Analyst", market_analyst)
graph_builder.add_node("Competitor Scout", competitor_scout)
graph_builder.add_node("Solution Designer", solution_designer)
graph_builder.add_node("Prioritization Planner", prioritization_planner)
graph_builder.add_node("PRD Writer", prd_writer)

graph_builder.add_edge("Problem Framer", "Market Analyst")
graph_builder.add_edge("Market Analyst", "Competitor Scout")
graph_builder.add_edge("Competitor Scout", "Solution Designer")
graph_builder.add_edge("Solution Designer", "Prioritization Planner")
graph_builder.add_edge("Prioritization Planner", "PRD Writer")
graph_builder.add_edge("PRD Writer", END)

pm_graph = graph_builder.compile()

# -------------------------------
# 4. Run Agent
# -------------------------------
def run_pm_agent(problem_statement: str, filename=None, csv_content=None, follow_up=None):
    if follow_up:
        # Direct LLM response for follow-up
        context = st.session_state.get("analysis_summary", "")
        prompt = f"Answer this question based on previous analysis:\n{follow_up}\n\nContext:\n{context}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"output": response.content, "history": prompt}

    # Initial run: execute graph
    final_state = pm_graph.invoke({"input": problem_statement})
    return final_state
