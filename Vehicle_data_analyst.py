import os
import streamlit as st
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain.agents import Tool, initialize_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

# 1. Setup Groq API Key
# os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_api_key = st.secrets["GROQ_API_KEY"]

# 2. Initialize Gemini LLM
llm = ChatGroq(model="Llama3-70b-8192",groq_api_key=groq_api_key)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 3. Agent Helper Functions

def simple_agent(name: str, prompt_template: str) -> Runnable:
    def _agent(state: dict) -> dict:
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

# 4. Define All Agents

problem_framer = simple_agent(
    "Problem Framer Agent",
    "Break the problem down into user impact, business goals, and tech implications. Format it as bullet points. Entire output should be within 500 words."
)

market_analyst = simple_agent(
    "Market Analyst Agent",
    "Search the web to identify relevant market trends, user behavior patterns, and industry stats. Entire output should be within 500 words."
)

competitor_scout = simple_agent(
    "Competitor Scout Agent",
    "Search the web to find top competitors and their features, pricing, and differentiation. Summarize insights. Entire output should be within 500 words."
)

solution_designer = simple_agent(
    "Solution Designer Agent",
    "Based on previous insights, suggest 2 product solutions with clear MVP scope. Entire output should be within 500 words."
)

prioritization_planner = simple_agent(
    "Prioritization Planner Agent",
    "Prioritize the features using RICE scoring and propose a phased roadmap: MVP and V1. Entire output should be within 500 words."
)

prd_writer = simple_agent(
    "PRD Writer Agent",
    "Write a Product Requirements Document with Overview, Objectives, Features, User Stories, and KPIs. Summarize the PRD within 1000 words."
)

# 5. Build LangGraph

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

# 6. Run the Graph

def run_pm_agent(problem_statement: str):
    print("\n Running Agentic PM for:", problem_statement)
    final_state = pm_graph.invoke({"input": problem_statement})
    print("\n Final Output:\n")
    print(final_state["output"])

    return final_state
