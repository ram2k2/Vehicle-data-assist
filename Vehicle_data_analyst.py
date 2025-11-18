import os
from data_tools import analyze_vehicle_data
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# The API Key is retrieved from environment variables (must be set in Streamlit Cloud secrets)
apiKey = os.environ.get("GEMINI_API_KEY", "") 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=apiKey)

@tool
def vehicle_data_parser(csv_content: str, filename: str) -> str:
    """
    Analyzes raw vehicle data (semicolon-delimited CSV) to calculate key metrics
    and returns a JSON string of the calculated summary metrics, including 'Raw Data' lists.
    Use this tool FIRST whenever a user uploads a new file or asks for a summary.
    """
    summary_dict = analyze_vehicle_data(csv_content, filename)
    # Convert the Python dictionary result to a string for the LLM to process
    return str(summary_dict)

tools = [vehicle_data_parser] 
system_prompt_template = """
You are an expert Data Analyst specializing in vehicle telemetry data.
Your goal is to provide insightful and conversational analysis of the provided data.
You MUST follow these rules:

1. **FIRST ACTION:** When a file is uploaded, you MUST call the `vehicle_data_parser` tool immediately to get the data summary.
2. **Summary:** Present the metrics in a structured, conversational format, ensuring all numerical values are **bolded** and include their appropriate units.
3. **Question Suggestion:** After the initial summary, always suggest at least 3 relevant questions the user could ask next (e.g., about trends, efficiency, or battery degradation).
4. **Footer:** For EVERY single response you generate, you MUST add a footer saying: "Data extracted from: [filename]".
5. **No Code:** Do not show the user any code or raw JSON output, only interpret and present the findings.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
