import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.schema.runnable import RunnableMap

# CSV Analysis Logic
def analyze_csv(file_path: str) -> str:
    try:
        # Auto-detect delimiter
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

# Runnable pipeline for Q&A
agent_chain = RunnableMap({
    "input": lambda x: x["input"],
    "response": prompt | llm
})

# Handle user query
def handle_query(user_query: str) -> str:
    try:
        result = agent_chain.invoke({"input": user_query})
        return result["response"].content
    except Exception as e:
        return f"Error: {str(e)}"
