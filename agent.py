import pandas as pd
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

# Initialize LLM (Gemini)
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Simple query handler
def handle_query(user_query: str) -> str:
    try:
        # Direct call to LLM without agent wrapper
        return llm.invoke(user_query)
    except Exception as e:
        return f"Error: {str(e)}"
