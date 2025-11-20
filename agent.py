import pandas as pd
from langchain.agents import AgentExecutor 
from langchain.agents.tool_calling_agent import create_tool_calling_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# CSV Analysis Logic
def analyze_csv(file_path: str) -> str:
    try:
        # Auto-detect delimiter (semicolon first, fallback to comma)
        try:
            df = pd.read_csv(file_path, sep=';', engine='python')
        except Exception:
            df = pd.read_csv(file_path, sep=',', engine='python')

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return "No numeric columns found for analysis."

        # Top 4 metrics by variance
        top_cols = df[numeric_cols].var().sort_values(ascending=False).head(4).index.tolist()
        summary_lines = [
            f"{col}: mean={df[col].mean():.2f}, max={df[col].max()}, min={df[col].min()}"
            for col in top_cols
        ]
        return "Summary of top metrics:\n" + "\n".join(summary_lines)
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"

# Tool for CSV analysis
tools = [
    Tool(
        name="CSV Analyzer",
        func=analyze_csv,
        description="Analyze uploaded CSV file and return insights"
    )
]

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional Vehicle Data Insights Assistant."),
    ("human", "{input}")
])

# Create agent using new API
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Handle user query
def handle_query(user_query: str) -> str:
    try:
        result = agent_executor.invoke({"input": user_query})
        return result["output"]
    except Exception as e:
        return f"Error: {str(e)}"
