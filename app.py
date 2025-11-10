# app.py
import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Vehicle Data Chat", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Data Chat Assistant")
st.caption("Upload a semicolon-delimited CSV and ask questions about it.")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload CSV
uploaded_file = st.file_uploader("Upload a semicolon-delimited CSV", type=["csv"])
if uploaded_file:
    try:
        content = uploaded_file.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(content), sep=";", engine="python", on_bad_lines="skip")
        df = df.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(",", ".", regex=False), errors="ignore"))
        st.session_state.df = df
        st.success(f"Loaded {uploaded_file.name} with {len(df):,} rows and {df.shape[1]} columns.")
        st.session_state.messages.append({"role": "assistant", "content": "CSV loaded. You can now ask questions like:\n- `summary`\n- `mean of engine_rpm`\n- `top 5 by vehicle_speed_kmh`"})
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask about your data…")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.df is None:
        reply = "Please upload a semicolon-delimited CSV first."
    else:
        df = st.session_state.df
        cols = df.columns.tolist()
        q = query.lower()

        if "summary" in q:
            reply = f"Rows: {len(df):,}\nColumns: {len(cols)}\nNumeric columns: {', '.join([c for c in cols if pd.api.types.is_numeric_dtype(df[c])])}"
        elif "columns" in q:
            reply = "Columns:\n- " + "\n- ".join(cols)
        else:
            found = False
            for col in cols:
                if col.lower() in q:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if "mean" in q:
                        reply = f"Mean of {col}: {s.mean():.2f}"
                        found = True
                    elif "median" in q:
                        reply = f"Median of {col}: {s.median():.2f}"
                        found = True
                    elif "max" in q:
                        reply = f"Max of {col}: {s.max():.2f}"
                        found = True
                    elif "min" in q:
                        reply = f"Min of {col}: {s.min():.2f}"
                        found = True
                    elif "std" in q or "standard deviation" in q:
                        reply = f"Standard deviation of {col}: {s.std():.2f}"
                        found = True
                    elif "missing" in q:
                        missing_pct = 100 * s.isna().mean()
                        reply = f"Missing values in {col}: {missing_pct:.1f}%"
                        found = True
                    break
            if not found:
                reply = "Sorry, I couldn't understand your question. Try asking about a column like `mean of engine_rpm`."

    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})