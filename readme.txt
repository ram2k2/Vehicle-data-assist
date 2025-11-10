Act like a product manager and suggest me how to build this solution. 
Lets start with a simple robust solution covering the requirements. 
Once it works fine, we can improve and add enhancement. 
My use case is, I will get a data extract of a vehicle user; 
I need to build a AI agent which will act as a data analyst that analyses the data
Provides useful insights to the user or a third party, with responses, 
Provides interesting insights, graphic visualizations and also interpretations using LLM. 
For example, the agent will give a summary of the extract 
Can suggest which questions can be asked
Always treat ';' as the only delimiter when parsing CSV files. Use manual parsing, not automatic.
For each response add a footer saying data extracted from (filename)

Summary:
- When summarizing, extract and clean values for:
  - "Total distance (km)"
  - "Fuel efficiency"
  - "High voltage battery State of Health (SOH)."
  - "Current vehicle speed."
- Remove invalid entries ("NV", "NA", empty).
- Calculate:
  - Total Distance = last - first value of "Total distance (km)"
  - Average Fuel Efficiency = mean of "Fuel efficiency"
  - Latest Battery SOH = last value of "High voltage battery State of Health (SOH)."
  - Average Vehicle Speed = mean of "Current vehicle speed."
- Present results in a structured format with units and bold formatting.

Usecases:
- Conversational
- Provides summary
- Suggest questions on its own
- free of Cost 
- Provide visualizations also
- use Gemini API key



Advancements:
- Work on multiple formats
