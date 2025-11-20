import pandas as pd
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import GoogleGenerativeAI

# CSV Analysis Logic
def analyze_csv(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path, sep=';', engine='python')  # Expect semicolon-delimited
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return "No numeric columns found for analysis."
        top_cols = df[numeric_cols].var().sort_values(ascending=False).head(4).index.tolist()
        summary_lines = [
            f"{col}: mean={df[col].mean():.2f}, max={df[col].max()}, min={df[col].min()}"
            for col in top_cols
        ]
        return "Summary of top metrics:\n" + "\n".join(summary_lines)
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"

# Define schema (optional for future use)
class QueryInput(BaseModel):
    query: str = Field(..., description="User query for vehicle data insights")

# Tools
tools = [Tool(name="CSV Analyzer", func=analyze_csv, description="Analyze uploaded CSV file and return insights")]

# LLM (Google Gemini)
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Initialize agent using old API
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

# Handle user query
def handle_query(user_query: str) -> str:
    try:
        return agent.run(user_query)
    except Exception as e:
        return f"Error: {str(e)}"
