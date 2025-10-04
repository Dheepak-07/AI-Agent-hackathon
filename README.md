# 🚀 Customer Feedback Prioritizer

A user-friendly web application built with Streamlit that uses a zero-shot classification AI model to automatically categorize and prioritize customer feedback from a CSV file. This tool is designed to help product teams quickly identify the most critical issues and feature requests.



---
## ✨ Features

* **Easy File Upload:** Upload your customer feedback data in CSV format directly through the web interface.
* **Dynamic Column Selection:** The app automatically detects all columns in your CSV, allowing you to easily select the one containing the feedback text.
* **AI-Powered Classification:** Leverages a Hugging Face Transformers model to classify each feedback entry into one of four categories: `Top Priority`, `Medium Priority`, `Least Priority`, or `feature required`.
* **Interactive Results:** View the processed data with the new 'Priority' column in a clean, interactive table.
* **Downloadable Output:** Download the complete, prioritized dataset as a new CSV file with a single click.

---
## 🛠️ Tech Stack

* **Python 3.9+**
* **Streamlit:** For the web application framework.
* **Pandas:** For data manipulation and handling CSV files.
* **Hugging Face Transformers:** For the core AI classification model.
* **PyTorch:** As the backend for the Transformers model.

---
## ⚙️ Setup and Local Installation

Follow these steps to run the application on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.9** or a newer version installed on your system.

### 2. Clone the Repository
Open your terminal and clone the project repository:
```bash
git clone https://github.com/Dheepak-07/AI-Agent-hackathon.git
cd AI-Agent-hackathon
```

### 3. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
<details> <summary><strong>macOS/Linux</strong></summary>
</details> <details> <summary><strong>Windows</strong></summary>
```bash
python -m venv venv
venv\Scripts\activate
```
</details>

### 4. Install Dependencies
Install all required Python packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run the App
Once setup is complete, run the app using:
```bash
streamlit run app.py
```

Your default browser will automatically open with the application running.
## 📋 Usage Steps:
Upload a CSV file using the sidebar panel.
Select the column containing raw feedback text.
Click "Analyze Feedback" to classify and prioritize.
View and download the results.
## ☁️ Deployment
This app is ready for Streamlit Community Cloud.
Create a public GitHub repository.
Push app.py and requirements.txt to the repo.
Go to https://share.streamlit.io, connect your GitHub, and select the repo to deploy.
## 📄 File Structure
```bash
├── app.py             # The main Streamlit app script
└── requirements.txt   # List of required Python packages
```
## 📬 Feedback or Contributions
Feel free to open issues or submit pull requests to contribute or report bugs.
