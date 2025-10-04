import os
import threading
import streamlit as st
import pandas as pd
import requests

# --- Page Setup ---
st.set_page_config(
    page_title="🚀 Feedback Prioritizer",
    page_icon="💬",
    layout="wide"
)

# --- Hugging Face API Setup ---
HF_MODEL = "typeform/distilbert-base-uncased-mnli"
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]  # store securely in .streamlit/secrets.toml
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def query_hf_zero_shot(text: str, labels: list) -> dict:
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels}
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"{response.status_code}: {response.text}"}

# --- Global State ---
if "model_ready" not in st.session_state:
    st.session_state["model_ready"] = True  # API-based, so always ready
if "model_error" not in st.session_state:
    st.session_state["model_error"] = None

# --- Streamlit UI ---
st.title("💬 Customer Feedback Prioritizer (Hugging Face API)")

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    feedback_col = None
    df = None

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            feedback_col = st.selectbox("Select the feedback column:", df.columns)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# --- Dynamic Model Status Indicator ---
if st.session_state["model_error"]:
    st.error(f"❌ Model API failed: {st.session_state['model_error']}")
else:
    st.success("✅ Hugging Face API ready!")

# --- Main App Workflow ---
if uploaded_file and feedback_col:
    st.subheader("📊 Ready to Analyze Feedback")

    analyze_button = st.button(
        "🚀 Start Analysis",
        type="primary"
    )

    if analyze_button:
        candidate_labels = ["Top Priority", "Medium Priority", "Least Priority", "Feature Required"]

        df = df.copy()
        df[feedback_col] = df[feedback_col].astype(str).fillna("")

        results = []
        progress = st.progress(0, text="Analyzing feedback...")
        total = len(df)

        for i, text in enumerate(df[feedback_col]):
            try:
                res = query_hf_zero_shot(text, candidate_labels)
                if "error" in res:
                    results.append("Error")
                else:
                    results.append(res["labels"][0])
            except Exception as e:
                results.append("Error")
            progress.progress((i + 1) / total, text=f"Processed {i + 1}/{total}")

        progress.empty()
        df["Priority"] = results

        st.session_state["processed_df"] = df

# --- Display Results ---
if "processed_df" in st.session_state:
    st.subheader("✅ Analysis Results")
    st.dataframe(st.session_state["processed_df"], use_container_width=True)

    csv_data = st.session_state["processed_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Prioritized Feedback",
        data=csv_data,
        file_name="prioritized_feedback.csv",
        mime="text/csv"
    )

# --- Footer ---
st.markdown("---")
st.caption("Built by Dheepak-07 | Optimized for Streamlit Cloud via Hugging Face API")