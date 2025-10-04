# app.py

# --- Required Library Versions ---
# This app was built and tested with the following library versions,
# which are also specified in the 'requirements.txt' file.
#
# streamlit
# pandas
# numpy
# torch==2.1.2
# torchvision==0.16.2
# torchaudio==2.1.2
# transformers==4.39.3
# accelerate
# ---------------------------------

import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Core AI and Data Functions ---

@st.cache_resource
def load_classifier():
    """
    Loads a lightweight zero-shot classification model.
    Using @st.cache_resource ensures this model is loaded only once.
    """
    st.write("Loading AI model for the first time... This may take a moment.")
    
    # Using a smaller, efficient model to fit within free cloud hosting limits.
    classifier = pipeline(
        "zero-shot-classification",
        model="Moritz/xtremedistil-l6-h256-mnli"
    )
    return classifier

def classify_feedback(feedback_list: list, classifier) -> list:
    """
    Classifies a list of feedback texts using the pre-loaded classifier.
    """
    if not feedback_list:
        return []

    candidate_labels = ["Top Priority", "Medium Priority", "Least Priority", "feature required"]
    
    # Show a progress bar for a better user experience
    progress_bar = st.progress(0, text="Analyzing feedback...")
    results = []
    
    # Process in batches to update the progress bar
    batch_size = 50 
    for i in range(0, len(feedback_list), batch_size):
        batch = feedback_list[i:i+batch_size]
        batch_results = classifier(batch, candidate_labels)
        results.extend(batch_results)
        progress = min((i + batch_size) / len(feedback_list), 1.0)
        progress_bar.progress(progress, text=f"Analyzing feedback... {int(progress*100)}% complete")

    progress_bar.empty() # Clear the progress bar
    
    # Extract just the top label for each piece of feedback
    top_priorities = [result['labels'][0] for result in results]
    return top_priorities

@st.cache_data
def convert_df_to_csv(df):
    """
    Helper function to convert the processed DataFrame to a downloadable CSV format.
    """
    return df.to_csv(index=False).encode('utf-8')

# --- Streamlit App Layout ---

st.set_page_config(page_title="Feedback Prioritizer", page_icon="🚀", layout="wide")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv']
    )
    
    # Controls are displayed only after a file is uploaded
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # 2. Column Selector
        feedback_column = st.selectbox(
            "Select the column with feedback text:",
            options=df.columns
        )
        
        # 3. Analysis Button
        analyze_button = st.button("Analyze Feedback", type="primary")

# --- Main Page ---
st.title("🚀 Customer Feedback Prioritizer")
st.write(
    "Upload a CSV file, select the column containing customer feedback, "
    "and this tool will automatically classify each entry by priority."
)

# Processing logic runs only when the button is clicked
if uploaded_file and analyze_button:
    st.subheader("📊 Analysis in Progress")
    
    # Load the model
    classifier = load_classifier()
    
    # Prepare and classify the data
    df_to_process = df.copy()
    df_to_process[feedback_column] = df_to_process[feedback_column].astype(str).fillna('')
    feedback_list = df_to_process[feedback_column].tolist()
    
    priorities_list = classify_feedback(feedback_list, classifier)
    
    # Add results to the DataFrame
    df_to_process['Priority'] = priorities_list
    
    # Store the processed DataFrame in session state to keep it available
    st.session_state['processed_df'] = df_to_process

# Display results and download button if processing is complete
if 'processed_df' in st.session_state:
    st.subheader("✅ Analysis Complete!")
    
    processed_df = st.session_state['processed_df']
    st.dataframe(processed_df)
    
    csv_data = convert_df_to_csv(processed_df)

    st.download_button(
       label="📥 Download Results as CSV",
       data=csv_data,
       file_name='prioritized_feedback.csv',
       mime='text/csv',
    )