
# 🚗 Vehicle Data Assist - Grand Version

An advanced Streamlit app powered by **LangChain**, **LangGraph**, and **Gemini LLM** for vehicle data analysis.

## ✅ Features
- Upload semicolon-delimited CSV files
- Auto-detect numeric columns
- Generate visualizations (histogram, trend, correlation heatmap)
- Summarize data and suggest proactive questions
- Conversational Q&A powered by Gemini

## 🔧 Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/vehicle-data-assist.git
   cd vehicle-data-assist

Final Functional Plan
Streamlit UI:
Upload CSV → Analyze and show summary + charts.
Chat interface → Ask questions (agent responds).

Agent:
Professional tone.
Uses Gemini via LangChain.
Has a tool for CSV analysis.

CSV Analysis:
Auto-detect delimiter.
Identify numeric columns.
Summarize top 4 metrics by variance.
Show interactive Plotly charts.

Core Features
Upload semicolon-delimited CSV files
Auto-detect delimiter (handle ; and fallback to ,).
Validate file type and structure.

Auto-detect numeric columns
Identify numeric columns dynamically.
Handle missing or mixed data gracefully.

Generate Visualizations
Histogram for each numeric column.
Trend chart (line plot for time or index vs key metric).
Correlation heatmap for numeric columns.

Summarize Data
Top 4 metrics by variance.
Mean, min, max for each.
Short structured summary.

Suggest Proactive Questions
Example: “Which metric shows the highest variability?”
“Is there a correlation between mileage and fuel consumption?”

Conversational Q&A powered by Gemini
LangChain agent integrated with Google Generative AI.
Professional, objective tone.
Can call CSV analysis tool when needed.

Version 1 – Simple & Functional
Features:
Upload semicolon-delimited CSV.
Auto-detect numeric columns.
Generate a basic summary (top 4 metrics by variance).
Simple Streamlit UI (upload + summary + chat).
Conversational Q&A powered by Gemini (basic integration).
Goal: Get a working MVP quickly.

Version 2 – Intermediate
Features:
Everything in V1.
Add basic visualizations:
Histogram for numeric columns.
Trend chart for one key metric.
Improve agent responses:
Suggest proactive questions after summary.
Handle missing data gracefully.
Goal: Make it visually appealing and slightly smarter.

Version 3 – Full Feature Set
Features:
Everything in V2.
Add correlation heatmap.
Advanced proactive question suggestions.
Persistent chat history.
Polished UI with tabs (Summary | Charts | Chat).
Goal: Complete your original vision.
