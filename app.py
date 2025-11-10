import os
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Optional LLM (Gemini) ---
GEMINI_AVAILABLE = False
try:
    # We require the SDK to be present to run the LLM router
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# --- LLM Helper ---
def _get_api_key():
    # Attempt to get API key from environment variable
    return os.getenv("GOOGLE_API_KEY")

def get_model_name() -> str | None:
    # Using gemini-2.5-flash for the largest context window capacity in the flash tier.
    return "gemini-2.5-flash"

# -----------------------------
# Page / Session
# -----------------------------
st.set_page_config(page_title="Vehicle Data Chat (LLM)", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Data Chat Assistant — LLM")
st.caption("Upload a semicolon-delimited CSV and ask questions freely. The LLM plans; the app computes locally.")

if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = None

# -----------------------------
# Footer & parsing helpers
# -----------------------------
def footer_text(filename: Optional[str]) -> str:
    name = filename if filename else "no file"
    return f"\n\n*data extracted from ({name})*"

def parse_semicolon_csv(raw: bytes) -> pd.DataFrame:
    """
    STRICT: semicolon is the ONLY delimiter; no auto-detection.
    """
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=";", engine="python", on_bad_lines="skip")
    return df

def coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Normalize comma-decimals in a column and coerce to numbers.
    (This is also where "NV", "NA", and empty strings become NaN, effectively cleaning the data)
    """
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def case_map(columns: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}

def resolve_col(name: str, columns: List[str]) -> Optional[str]:
    """Resolves a column name (case-insensitive) based on a candidate name."""
    if name in columns:
        return name
    return case_map(columns).get(name.lower())


def calculate_summary(df: pd.DataFrame, filename: str) -> str:
    """
    Calculates the specific summary metrics requested by the user.
    Handles data cleaning (NV, NA, empty) via coerce_numeric.
    """
    
    # Define required column candidates (resilient to km/miles and minor naming changes)
    COL_MAPPING = {
        "Distance": ["Total distance (km)", "Total distance (miles)"],
        "FuelEfficiency": ["Fuel efficiency"],
        "SOH": ["High voltage battery State of Health (SOH)."],
        "Speed": ["Current vehicle speed."],
    }

    all_cols = df.columns.tolist()
    resolved_cols = {}
    
    # Resolve columns using the existing robust helper and candidate lists
    for metric, candidates in COL_MAPPING.items():
        resolved_cols[metric] = None
        for name in candidates:
            # resolve_col handles case-insensitivity
            r_col = resolve_col(name, all_cols)
            if r_col:
                # Store the resolved name and the original candidate name to infer units
                resolved_cols[metric] = {"name": r_col, "unit_source": name}
                break
                
    # Extract resolved column names and check for missing mandatory columns
    missing = [metric for metric, r in resolved_cols.items() if r is None]
    
    if missing:
        # Create a list of the *intended* column names that were missing
        missing_names = [COL_MAPPING[m][0] for m in missing]
        return f"❌ **Error:** Cannot calculate summary. The following required data points were not found (expected one of these columns): {', '.join(missing_names)}." + footer_text(filename)
            
    # Extract resolved column variables for cleaner use
    r_dist = resolved_cols["Distance"]["name"]
    # Determine the unit based on the column name found
    unit_dist = "km" if "(km)" in resolved_cols["Distance"]["unit_source"].lower() else "miles"
    r_fuel = resolved_cols["FuelEfficiency"]["name"]
    r_soh = resolved_cols["SOH"]["name"]
    r_speed = resolved_cols["Speed"]["name"]
    
    results = []

    # 1. Total Distance Traveled ({unit_dist}) = last - first
    # Dropna removes all invalid entries (including 'NV', 'NA', empty, and non-numeric)
    s_dist = coerce_numeric(df[r_dist]).dropna()
    if len(s_dist) >= 2:
        total_distance = s_dist.iloc[-1] - s_dist.iloc[0]
        # Use the dynamically determined unit label
        results.append(f"**Total Distance Traveled:** {total_distance:,.2f} {unit_dist}")
    else:
        results.append(f"**Total Distance Traveled:** Insufficient data (found {len(s_dist)} valid points)")

    # 2. Average Fuel Efficiency = mean
    s_fuel = coerce_numeric(df[r_fuel]).dropna()
    if len(s_fuel) > 0:
        avg_fuel = s_fuel.mean()
        # Note: Unit not provided in request, using a generic label
        results.append(f"**Average Fuel Efficiency:** {avg_fuel:,.2f}")
    else:
        results.append("**Average Fuel Efficiency:** No valid data")

    # 3. Latest Battery SOH = last value
    s_soh = coerce_numeric(df[r_soh]).dropna()
    if len(s_soh) > 0:
        latest_soh = s_soh.iloc[-1]
        results.append(f"**Latest Battery SOH:** {latest_soh:,.2f}%")
    else:
        results.append("**Latest Battery SOH:** No valid data")

    # 4. Average Vehicle Speed = mean
    s_speed = coerce_numeric(df[r_speed]).dropna()
    if len(s_speed) > 0:
        avg_speed = s_speed.mean()
        # Using a generic unit for speed based on context
        results.append(f"**Average Vehicle Speed:** {avg_speed:,.2f} units/h") 
    else:
        results.append("**Average Vehicle Speed:** No valid data")
        
    
    summary_text = "\n".join(results)
    
    return f"""
## 📊 Vehicle Data Summary Report
{summary_text}
---
*Note: Invalid entries ('NV', 'NA', empty strings, and comma-decimals) were automatically removed before calculation.*
""" + footer_text(filename)


# -----------------------------
# LLM Router & Narrator
# -----------------------------
ALLOWED_OPS = {"shape", "columns", "aggregate", "filter", "sort", "head"}
ALLOWED_METRICS = {"mean", "median", "min", "max", "std", "sum", "count", "missing_pct"}

def get_model() -> Optional[Any]:
    api_key = _get_api_key()
    if not (GEMINI_AVAILABLE and api_key):
        return None
    genai.configure(api_key=api_key)
    # Using the highly capable gemini-2.5-flash model
    return genai.GenerativeModel(get_model_name())

def router_instruction() -> str:
    return """
You are a router that converts a user's question about a tabular CSV into a SMALL JSON plan.

ALLOWED OPS:
- shape: {}
- columns: {}
- aggregate: {"metrics":[{"agg":"mean|median|min|max|std|sum|count|missing_pct","col":"<exact header>","as":"<optional>"}]}
- filter: {"col":"<exact header>","op":"==|!=|>|>=|<|<=","value":"<number|string>"}
- sort: {"by":"<exact header>","ascending": true|false}
- head: {"n": <int>}

SCHEMA:
{"plan":[{"op":"<one_of_allowed_ops>", ...args}], "return":"text|table|both"}

RULES:
- Use ONLY the columns provided (case sensitive, exact).
- Use ONLY allowed ops/fields. Keep the plan short and safe.
- For “top N by X”: use sort(desc) + head(N).
- For “bottom N by X”: use sort(asc) + head(N).
- If asking “how many rows/columns”: use shape.
- If asking “headers”: use columns.
- If “driving behaviour/behavior” or “overview”: choose sensible aggregates on speed/load/efficiency if such columns exist (names must match the provided headers exactly).
- Output ONLY minified JSON, no prose.
"""

def route_with_llm(question: str, columns: List[str]) -> Dict[str, Any]:
    model = get_model()
    if model is None:
        raise RuntimeError("LLM router unavailable. Set GOOGLE_API_KEY and install google-generativeai.")
    prompt = router_instruction() + "\n" + json.dumps({"columns": columns, "question": question})
    
    # Instantiate the model with JSON output configuration
    model_json = genai.GenerativeModel(
        get_model_name(),
        generation_config={"response_mime_type": "application/json"}
    )
    resp = model_json.generate_content(prompt)
    text = (getattr(resp, "text", "") or "").strip()
    try:
        plan = json.loads(text)
        if not isinstance(plan, dict) or "plan" not in plan:
            raise ValueError("Invalid router JSON.")
        return plan
    except Exception as e:
        # Provide more context on the failure
        raise RuntimeError(f"Router produced invalid JSON: {e}\nRaw: {text[:400]}")

def narrate_with_llm(question: str, facts_text: str) -> Optional[str]:
    """
    Optional natural-language answer grounded on computed facts.
    """
    model = get_model()
    if model is None:
        return None
    prompt = f"""You are an analyst. Answer the user's question concisely using ONLY these computed facts.

Question:
{question}

Computed facts (tabular/text you can quote):
{facts_text}

Rules:
- Be precise and neutral. Do not fabricate numbers.
- If facts don't fully answer, say what else is needed.
- Keep to 4-7 short sentences.
- IMPORTANT: Conclude your response by proactively asking the user a single, relevant, follow-up question based on the data. For example: 'Would you like to know the maximum value for [Specific Column Name]?' or 'How did the average [Other Column Name] change over this period?'
"""
    try:
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception:
        # Fail silently if narration fails, just return the facts
        return None

# -----------------------------
# Validate plan
# -----------------------------
def validate_plan(plan: Dict[str, Any], df: pd.DataFrame, max_rows: int = 20) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(plan, dict) or "plan" not in plan or not isinstance(plan["plan"], list):
        return False, "Plan format invalid.", plan

    columns = df.columns.tolist()
    normalized = {"plan": [], "return": plan.get("return")}

    for step in plan["plan"]:
        if not isinstance(step, dict) or "op" not in step:
            return False, "Each step must be an object with 'op'.", plan
        op = step["op"]
        if op not in ALLOWED_OPS:
            return False, f"Unsupported operation: {op}", plan

        stp = {"op": op}

        if op in ("shape", "columns"):
            pass

        elif op == "aggregate":
            metrics = step.get("metrics", [])
            if not isinstance(metrics, list) or not metrics:
                return False, "aggregate.metrics must be a non-empty list.", plan
            out = []
            for m in metrics:
                if not isinstance(m, dict):
                    return False, "aggregate metric must be an object.", plan
                agg = m.get("agg"); col = m.get("col"); alias = m.get("as")
                if agg not in ALLOWED_METRICS:
                    return False, f"Unsupported metric: {agg}", plan
                rcol = resolve_col(col, columns)
                if not rcol:
                    return False, f"Column not found: {col}", plan
                out.append({"agg": agg, "col": rcol, "as": alias})
            stp["metrics"] = out

        elif op == "filter":
            col = step.get("col"); op2 = step.get("op"); val = step.get("value")
            if not (col and op2 and (val is not None)):
                return False, "filter requires 'col','op','value'.", plan
            if op2 not in ("==", "!=", ">", ">=", "<", "<="):
                return False, f"Unsupported operator: {op2}", plan
            rcol = resolve_col(col, columns)
            if not rcol:
                return False, f"Column not found: {col}", plan
            stp.update({"col": rcol, "op": op2, "value": val})

        elif op == "sort":
            by = step.get("by"); asc = step["ascending"]
            if not by:
                return False, "sort requires 'by'.", plan
            rcol = resolve_col(by, columns)
            if not rcol:
                return False, f"Column not found: {by}", plan
            stp.update({"by": rcol, "ascending": bool(step.get("ascending", False))})

        elif op == "head":
            n = step.get("n", 5)
            try: n = int(n)
            except: n = 5
            n = max(1, min(max_rows, n))
            stp["n"] = n

        normalized["plan"].append(stp)

    ret = normalized.get("return")
    if ret and ret not in ("text", "table", "both"): normalized["return"] = "both"
    return True, "ok", normalized

# -----------------------------
# Execute plan (local, private)
# -----------------------------
def execute_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[str, Optional[pd.DataFrame]]:
    lines: List[str] = []
    table: Optional[pd.DataFrame] = None
    working = df.copy()

    for step in plan["plan"]:
        op = step["op"]

        if op == "shape":
            lines.append(f"Rows: {len(working):,}, Columns: {working.shape[1]}")

        elif op == "columns":
            cols = working.columns.tolist()
            lines.append("Headers: " + ", ".join(cols))

        elif op == "filter":
            col, op2, val = step["col"], step["op"], step["value"]
            s_num = coerce_numeric(working[col])
            # decide numeric or string compare
            val_num = None
            try:
                val_num = float(str(val).replace(",", "."))
            except Exception:
                val_num = None
            if val_num is not None and (pd.api.types.is_numeric_dtype(working[col]) or s_num.notna().any()):
                left, right = s_num, val_num
            else:
                left, right = working[col].astype(str), str(val)

            if op2 == "==": mask = left == right
            elif op2 == "!=": mask = left != right
            elif op2 == ">": mask = left > right
            elif op2 == ">=": mask = left >= right
            elif op2 == "<": mask = left < right
            elif op2 == "<=": mask = left <= right
            working = working[mask]

        elif op == "sort":
            by = step["by"]; asc = step["ascending"]
            s_num = coerce_numeric(working[by])
            if s_num.notna().any():
                working = working.assign(_k=s_num).sort_values("_k", ascending=asc).drop(columns=["_k"])
            else:
                working = working.sort_values(by, ascending=asc, kind="mergesort")

        elif op == "head":
            working = working.head(step["n"])
            table = working

        elif op == "aggregate":
            for m in step["metrics"]:
                agg, col, alias = m["agg"], m["col"], m.get("as")
                s = coerce_numeric(working[col])
                if agg == "missing_pct":
                    val = float(100 * s.isna().mean())
                elif agg == "count":
                    val = int(s.notna().sum())
                else:
                    func = {"mean": np.nanmean, "median": np.nanmedian, "min": np.nanmin,
                            "max": np.nanmax, "std": np.nanstd, "sum": np.nansum}[agg]
                    val = float(func(s))
                
                name = alias or f"{agg}({col})"
                
                # Enhanced formatting for aggregates
                if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                    if isinstance(val, float) and val >= 10:
                        lines.append(f"{name}: {val:,.2f}") # Two decimal places for large floats
                    elif isinstance(val, float) and val < 10 and val > 0:
                        lines.append(f"{name}: {val:.4g}") # Use general format for smaller floats
                    else:
                        lines.append(f"{name}: {val:,}")
                else:
                    lines.append(f"{name}: —")

    ret = plan.get("return", "both")
    if ret == "text":
        return ("\n".join(lines) if lines else "Done."), None
    elif ret == "table":
        return ("", table if table is not None else working.head(20))
    else:
        return ("\n".join(lines)), (table if table is not None else None)

# -----------------------------
# Upload (no Test.csv; strict ;)
# -----------------------------
uploaded = st.file_uploader("Upload a **semicolon-delimited** CSV", type=["csv"])
if uploaded is not None:
    try:
        raw = uploaded.read()
        raw_df = parse_semicolon_csv(raw)
        st.session_state.raw_df = raw_df.copy()
        st.session_state.df = raw_df  # keep original; coercion happens per-op
        st.session_state.filename = uploaded.name
        
        st.success(f"Loaded {uploaded.name} with {len(raw_df):,} rows and {raw_df.shape[1]} columns.")
        
        # Display headers in an expander for user reference
        with st.expander("View available column headers"):
            st.write(", ".join(raw_df.columns.tolist()))

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "CSV loaded. Ask anything (e.g., 'how is the driving behaviour', "
                "'top 5 by vehicle_speed_kmh', 'mean of engine_rpm', 'filter engine_temp_c > 95', "
                "**'show summary'**)."
            ) + footer_text(st.session_state.filename)
        })
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

# -----------------------------
# Chat history
# -----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "_df" in m and isinstance(m["_df"], pd.DataFrame):
            st.dataframe(m["_df"], use_container_width=True)
        if "_plan" in m and m["_plan"]:
            with st.expander("Show LLM Plan (JSON)"):
                st.code(json.dumps(m["_plan"], indent=2), language="json")


# -----------------------------
# Chat input
# -----------------------------
q = st.chat_input("Ask about your data…", disabled=(st.session_state.df is None))
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)

    if st.session_state.df is None:
        reply = "Please upload a semicolon-delimited CSV first." + footer_text(None)
        with st.chat_message("assistant"): st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        df = st.session_state.df
        cols = df.columns.tolist()

        # --- NEW: Summary Check (Bypasses LLM for deterministic calculation) ---
        q_lower = q.lower()
        if "summary" in q_lower or "report" in q_lower or "metrics" in q_lower:
            with st.spinner("Calculating custom summary metrics..."):
                summary_output = calculate_summary(df, st.session_state.filename)
                
            with st.chat_message("assistant"):
                st.markdown(summary_output)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": summary_output
            })
            
        # --- END NEW: Summary Check ---

        # If it's not a summary request, proceed with LLM routing
        elif not (GEMINI_AVAILABLE and _get_api_key()):
            msg = (
                "LLM routing is disabled. Set **GOOGLE_API_KEY** (Gemini) and restart the app."
                + footer_text(st.session_state.filename)
            )
            with st.chat_message("assistant"): st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            try:
                # ROUTE
                plan = route_with_llm(q, cols)
                ok, msg, norm_plan = validate_plan(plan, df)
                
                if not ok:
                    text = f"Couldn't validate your request: {msg}" + footer_text(st.session_state.filename)
                    with st.chat_message("assistant"): st.markdown(text)
                    st.session_state.messages.append({"role": "assistant", "content": text, "_plan": plan})
                else:
                    # EXECUTE
                    facts_text, table = execute_plan(df, norm_plan)

                    # NARRATE
                    narrative = narrate_with_llm(q, facts_text) or ""
                    answer = (narrative.strip() if narrative else facts_text) + footer_text(st.session_state.filename)

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if isinstance(table, pd.DataFrame) and not table.empty:
                            st.dataframe(table, use_container_width=True)
                        
                        # Display the plan
                        with st.expander("Show LLM Plan (JSON)"):
                            st.code(json.dumps(plan, indent=2), language="json")

                    record = {"role": "assistant", "content": answer, "_plan": plan}
                    if isinstance(table, pd.DataFrame) and not table.empty:
                        record["_df"] = table
                    st.session_state.messages.append(record)

            except Exception as e:
                err = f"Sorry, I couldn't process that: {e}" + footer_text(st.session_state.filename)
                with st.chat_message("assistant"): st.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})