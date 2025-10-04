# app.py
# --- Required Library Versions ---
# Tested with:
# streamlit
# pandas
# numpy
# torch==2.1.2
# torchvision==0.16.2
# torchaudio==2.1.2
# transformers==4.39.3
# accelerate

import os
import streamlit as st
import pandas as pd
from transformers import pipeline
from huggingface_hub import login

# --- (Optional) Hugging Face Authentication ---
# If your model is private, set HF_TOKEN in Streamlit Secrets or Environment Variable
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
if HF_TOKEN:
    login(HF_TOKEN)

# --- Streamlit Page Config ---
st.set_page_config(page_title="🚀 Feedback Prioritizer", page_icon="💬", layout="wide")

# --- Core AI Functionality ---

@st.cache_resource
def load_classifier():
    """
    Loads a lightweight zero-shot classification model for feedback prioritization.
    Cached to prevent reloading on every rerun.
    """
    with st.spinner("🚀 Initializing AI model... Please wait ~30 seconds."):
        classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",  # Smaller and faster model
            device_map="auto",  # Automatically picks CPU/GPU if available
        )
    return classifier


def classify_feedback(feedback_list, classifier):
    """
    Classifies customer feedback into priority levels using the model.
    """
    if not feedback_list:
        return []

    candidate_labels = ["Top Priority", "Medium Priority", "Least Priority", "Feature Required"]
    
    progress_bar = st.progress(0, text="Analyzing feedback...")
    results = []
    batch_size = 20  # Smaller batches for low-memory environments

    for i in range(0, len(feedback_list), batch_size):
        batch = feedback_list[i:i + batch_size]
        try:
            batch_results = classifier(batch, candidate_labels)
        except Exception as e:
            st.error(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            continue

        # Handle both single and multi-result return formats
        if isinstance(batch_results, dict):
            results.append(batch_results)
        else:
            results.extend(batch_results)

        progress = min((i + batch_size) / len(feedback_list), 1.0)
        progress_bar.progress(progress, text=f"Analyzing feedback... {int(progress * 100)}%")

    progress_bar.empty()

    # Extract top label per entry
    return [r['labels'][0] for r in results if 'labels' in r]


@st.cache_data
def convert_df_to_csv(df):
    """
    Converts a DataFrame to downloadable CSV bytes.
    """
    return df.to_csv(index=False).encode('utf-8')


# --- Streamlit App Layout ---
st.title("🚀 Customer Feedback Prioritizer")
st.write(
    """
    Upload a CSV file containing customer feedback, and this app will automatically
    categorize each feedback entry by **priority** using advanced NLP.
    """
)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            st.stop()

        feedback_column = st.selectbox("Select the column with feedback text:", options=df.columns)
        analyze_button = st.button("Analyze Feedback", type="primary")
    else:
        df, feedback_column, analyze_button = None, None, None


# --- Main Logic ---
if uploaded_file and analyze_button:
    st.subheader("📊 Running Analysis...")

    # Load AI model
    classifier = load_classifier()

    # Preprocess feedback data
    df_to_process = df.copy()
    df_to_process[feedback_column] = df_to_process[feedback_column].astype(str).fillna('')

    # Classify
    priorities = classify_feedback(df_to_process[feedback_column].tolist(), classifier)
    df_to_process["Priority"] = priorities

    # Save results
    st.session_state["processed_df"] = df_to_process

# --- Results Display ---
if "processed_df" in st.session_state:
    st.subheader("✅ Analysis Complete!")
    processed_df = st.session_state["processed_df"]

    st.dataframe(processed_df, use_container_width=True)

    csv_data = convert_df_to_csv(processed_df)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="prioritized_feedback.csv",
        mime="text/csv"
    )

# --- Footer ---
st.markdown("---")
st.caption("Powered by 🤖 Hugging Face Transformers + Streamlit | Developed by Coder")
