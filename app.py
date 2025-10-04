import os
import threading
import streamlit as st
import pandas as pd
import torch
from transformers import pipeline


# --- Page Setup ---
st.set_page_config(
    page_title="🚀 Feedback Prioritizer",
    page_icon="💬",
    layout="wide"
)

# --- Global State ---
if "model_ready" not in st.session_state:
    st.session_state["model_ready"] = False
if "classifier" not in st.session_state:
    st.session_state["classifier"] = None
if "model_error" not in st.session_state:
    st.session_state["model_error"] = None

# --- Background Model Loader ---
def load_model_async():
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device_map="auto",
            model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
        )
        st.session_state["classifier"] = classifier
        st.session_state["model_ready"] = True
    except Exception as e:
        st.session_state["model_error"] = str(e)

# --- Trigger background load on first run ---
if "model_thread_started" not in st.session_state:
    threading.Thread(target=load_model_async, daemon=True).start()
    st.session_state["model_thread_started"] = True

# --- Streamlit UI ---
st.title("💬 Customer Feedback Prioritizer")
st.caption("Powered by 🤖 Transformers — Classify and prioritize feedback efficiently.")

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
    st.error(f"❌ Model failed to load: {st.session_state['model_error']}")
elif not st.session_state["model_ready"]:
    with st.spinner("⚡ Loading AI model in background..."):
        st.info("You can upload your CSV while the model loads.")
else:
    st.success("✅ AI model loaded and ready!")

# --- Main App Workflow ---
if uploaded_file and feedback_col:
    st.subheader("📊 Ready to Analyze Feedback")

    analyze_button = st.button(
        "🚀 Start Analysis",
        type="primary",
        disabled=not st.session_state["model_ready"]
    )

    if analyze_button and st.session_state["model_ready"]:
        classifier = st.session_state["classifier"]
        candidate_labels = ["Top Priority", "Medium Priority", "Least Priority", "Feature Required"]

        df = df.copy()
        df[feedback_col] = df[feedback_col].astype(str).fillna("")

        results = []
        progress = st.progress(0, text="Analyzing feedback...")
        total = len(df)

        for i, text in enumerate(df[feedback_col]):
            try:
                res = classifier(text, candidate_labels)
                results.append(res["labels"][0])
            except Exception:
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
st.caption("Built by Coder | Optimized for Streamlit Cloud | GPT-5 Guidance")
