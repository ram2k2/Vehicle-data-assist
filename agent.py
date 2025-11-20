import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

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

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Handle user query
def handle_query(user_query: str) -> str:
    try:
        messages = [
            SystemMessage(content="You are a professional Vehicle Data Insights Assistant."),
            HumanMessage(content=user_query)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"
