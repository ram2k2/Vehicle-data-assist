import streamlit as st
import os
from google import genai
from google.genai import types

# --- Configuration ---
# Your API key will be securely read from the environment variables managed by the platform.
# No need to hardcode the key.
try:
    API_KEY = os.environ['GEMINI_API_KEY']
except KeyError:
    # Fallback message if running locally without environment variable set
    API_KEY = None 
    st.error("GEMINI_API_KEY environment variable not found. Please set it to run the app.")

# Initialize the Gemini Client
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        client = None
else:
    client = None

MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'
SYSTEM_INSTRUCTION = "You are a concise, helpful, and friendly chat assistant optimized for Streamlit apps. Keep your answers brief."

# --- Session State Management ---

if "chat_history" not in st.session_state:
    # Initialize chat history with the system instruction
    # FIXED: Using types.Part(text=...) instead of the from_text class method for robustness.
    st.session_state.chat_history = [
        types.Content(
            role="model", 
            parts=[types.Part(text="Hello! I'm your minimal Streamlit Gemini Assistant. How can I help you today?")]
        )
    ]

# --- Core Functions ---

def send_message(prompt):
    """Handles sending the user message and getting a response from Gemini."""
    if not client:
        return # Skip if client failed to initialize

    # 1. Add user message to history
    st.session_state.chat_history.append(
        types.Content(role="user", parts=[types.Part(text=prompt)]) # FIX applied here
    )
    
    # 2. Configure the API call
    # We pass the full history (excluding the first model message for context)
    # The system instruction is handled by the client configuration, not in the chat history.
    contents_to_send = st.session_state.chat_history
    
    try:
        # 3. Call the Gemini API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents_to_send,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )
        )

        # 4. Extract text and add model response to history
        model_text = response.text
        st.session_state.chat_history.append(
            types.Content(role="model", parts=[types.Part(text=model_text)]) # FIX applied here
        )
        
    except Exception as e:
        error_message = f"Error: Could not get response from Gemini. ({e})"
        st.session_state.chat_history.append(
            types.Content(role="model", parts=[types.Part(text=error_message)]) # FIX applied here
        )
        st.error(error_message)


# --- Streamlit UI Layout ---

st.set_page_config(page_title="Lean Streamlit Chat", layout="centered")
st.title("💬 Lean Gemini Streamlit Chat")

# Display chat messages from history
for message in st.session_state.chat_history:
    # Exclude the system instruction that was used to initialize the history
    if message.role != "user" and message.parts[0].text.startswith("Hello! I'm your minimal Streamlit Gemini Assistant"):
        # Display the first welcome message
        with st.chat_message("assistant"):
            st.write(message.parts[0].text)
        continue

    if message.role == "user":
        # Streamlit handles the 'user' role
        with st.chat_message("user"):
            st.write(message.parts[0].text)
    else:
        # Streamlit handles the 'model' role as 'assistant'
        with st.chat_message("assistant"):
            st.write(message.parts[0].text)


# Input text box for the user
if prompt := st.chat_input("Say something..."):
    send_message(prompt)