# agent.py
import os
import pandas as pd
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langgraph.graph import Graph

def create_agent(df):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    google_api_key=os.getenv("GEMINI_API_KEY")  # Reads from secrets
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def query_csv(query: str) -> str:
        try:
            return str(df.query(query))
        except Exception:
            return "Could not process query. Try a different format."

    def summarize_data(_: str) -> str:
        return str(df.describe().to_dict())

    tools = [
        Tool(name="CSVQuery", func=query_csv, description="Query the CSV using pandas syntax."),
        Tool(name="Summarizer", func=summarize_data, description="Summarize the dataset.")
    ]

    return initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)

def create_workflow(df):
    graph = Graph()

    def parse_csv():
        return df

    def validate_columns():
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        return {"numeric_cols": list(numeric_cols), "all_cols": list(df.columns)}

    def summarize_data():
        return df.describe().to_dict()

    def suggest_questions(numeric_cols):
        questions = []
        if len(numeric_cols) > 0:
            questions.append(f"What is the average {numeric_cols[0]}?")
            questions.append(f"Which record has the highest {numeric_cols[0]}?")
            if len(numeric_cols) > 1:
                questions.append(f"Is there a correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
        if 'Model' in df.columns:
            questions.append("Which model performs best based on key metrics?")
        return questions

    graph.add_node("parse_csv", parse_csv)
    graph.add_node("validate_columns", validate_columns)
    graph.add_node("summarize", summarize_data)
    graph.add_node("suggest_questions", lambda: suggest_questions(validate_columns()["numeric_cols"]))

    graph.add_edge("parse_csv", "validate_columns")
    graph.add_edge("validate_columns", "summarize")
    graph.add_edge("validate_columns", "suggest_questions")

    return graph
