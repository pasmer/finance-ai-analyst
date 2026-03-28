# Introduction
  Last month, I found myself staring at my bank statement, trying to figure out where my money was actually going. Spreadsheets felt cumbersome. Existing apps are like black boxes, and the worst part is that they demand I upload my sensitive financial data to a cloud server. I wanted something different. I wanted an AI data analyst that could analyze my spending, spot unusual transactions, and give me clear insights — all while keeping my data 100% local. So, I built one.

 Preview the Agenda—Plan Your Summit Journey!
What started as a weekend project turned into a deep dive into real-world data preprocessing, practical machine learning, and the power of local large language models (LLMs). In this article, I will walk you through how I created an AI-powered financial analysis app using Python with "Vibe Coding." Along the way, you will learn many practical concepts that apply to any data science project, whether you are analyzing sales logs, sensor data, or customer feedback.

 Apply Now—Unlock Exclusive CDAO Circle Access!
By the end, you will understand:
	•	How to build a robust data preprocessing pipeline that handles messy, real-world CSV files
	•	How to choose and implement machine learning models when you have limited training data
	•	How to design interactive visualizations that actually answer user questions
	•	How to integrate a local LLM for generating natural-language insights without sacrificing privacy
The complete source code is available on GitHub. Feel free to fork it, extend it, or use it as a starting point for your own AI data analyst.
 

Fig. 1: App dashboard showing spending breakdown and AI insights | Image by Author

 


# The Problem: Why I Built This
  Most personal finance apps share a fundamental flaw: your data leaves your control. You upload bank statements to services that store, process, and potentially monetize your information. I wanted a tool that:
	1.	Let me upload and analyze data instantly
	2.	Processed everything locally — no cloud, no data leaks
	3.	Provided AI-powered insights, not just static charts
This project became my vehicle for learning several concepts that every data scientist should know, like handling inconsistent data formats, selecting algorithms that work with small datasets, and building privacy-preserving AI features.
 

# Project Architecture
  Before diving into code, here is a project structure showing how the pieces fit together:
 
project/   
  ├── app.py              # Main Streamlit app
  ├── config.py           # Settings (categories, Ollama config)
  ├── preprocessing.py    # Auto-detect CSV formats, normalize data
  ├── ml_models.py        # Transaction classifier + Isolation Forest anomaly detector
  ├── visualizations.py   # Plotly charts (pie, bar, timeline, heatmap)
  ├── llm_integration.py  # Ollama streaming integration
  ├── requirements.txt    # Dependencies
  ├── README.md           # Documentation with "deep dive" lessons
  └── sample_data/
    ├── sample_bank_statement.csv
    └── sample_bank_format_2.csv
 
We will look at building each layer step by step.
 

# Step 1: Building a Robust Data Preprocessing Pipeline
  The first lesson I learned was that real-world data is messy. Different banks export CSVs in completely different formats. Chase Bank uses "Transaction Date" and "Amount." Bank of America uses "Date," "Payee," and separate "Debit"/"Credit" columns. Moniepoint and OPay each have their own styles.
A preprocessing pipeline must handle these differences automatically.
 
// Auto-Detecting Column Mappings
I built a pattern-matching system that identifies columns regardless of naming conventions. Using regular expressions, we can map unclear column names to standard fields.
import re

COLUMN_PATTERNS = {
    "date": [r"date", r"trans.*date", r"posting.*date"],
    "description": [r"description", r"memo", r"payee", r"merchant"],
    "amount": [r"^amount$", r"transaction.*amount"],
    "debit": [r"debit", r"withdrawal", r"expense"],
    "credit": [r"credit", r"deposit", r"income"],
}

def detect_column_mapping(df):
    mapping = {}
    for field, patterns in COLUMN_PATTERNS.items():
        for col in df.columns:
            for pattern in patterns:
                if re.search(pattern, col.lower()):
                    mapping[field] = col
                    break
    return mapping
 
The key insight: design for differences, not specific formats. This approach works for any CSV that uses common financial terms.
 
// Normalizing to a Standard Schema
Once columns are detected, we normalizeeverything into a consistent structure. For example, banks that split debits and credits need to be combined into a single amount column (negative for expenses, positive for income):
if "debit" in mapping and "credit" in mapping:
    debit = df[mapping["debit"]].apply(parse_amount).abs() * -1
    credit = df[mapping["credit"]].apply(parse_amount).abs()
    normalized["amount"] = credit + debit
 
Key takeaway: Normalize your data as soon as possible. It simplifies every following operation, like feature engineering, machine learning modeling, and visualization.
 

Fig 2: The preprocessing report shows what the pipeline detected, giving users transparency | Image by Author

 


# Step 2: Choosing Machine Learning Models for Limited Data
  The second major challenge is limited training data. Users upload their own statements, and there is no massive labeled dataset to train a deep learning model. We need algorithms that work well with small samples and can be augmented with simple rules.
 
// Transaction Classification: A Hybrid Approach
Instead of pure machine learning, I built a hybrid system:
	1.	Rule-based matching for confident cases (e.g., keywords like "WALMART" → groceries)
	2.	Pattern-based fallback for ambiguous transactions
SPENDING_CATEGORIES = {
    "groceries": ["walmart", "costco", "whole foods", "kroger"],
    "dining": ["restaurant", "starbucks", "mcdonald", "doordash"],
    "transportation": ["uber", "lyft", "shell", "chevron", "gas"],
    # ... more categories
}

def classify_transaction(description, amount):
    for category, keywords in SPENDING_CATEGORIES.items():
        if any(kw in description.lower() for kw in keywords):
            return category
    return "income" if amount > 0 else "other"
 
This approach works immediately without any training data, and it is easy for users to understand and customize.
 
// Anomaly Detection: Why Isolation Forest?
For detecting unusual spending, I needed an algorithm that could:
	1.	Work with small datasets (unlike deep learning)
	2.	Make no assumptions about data distribution (unlike statistical methods like Z-score alone)
	3.	Provide fast predictions for an interactive UI
Isolation Forest from scikit-learn ticked all the boxes. It isolates anomalies by randomly partitioning the data. Anomalies are few and different, so they require fewer splits to isolate.
from sklearn.ensemble import IsolationForest

detector = IsolationForest(
    contamination=0.05,  # Expect ~5% anomalies
    random_state=42
)
detector.fit(features)
predictions = detector.predict(features)  # -1 = anomaly
 
I also combined this with simple Z-score checks to catch obvious outliers. A Z-scoredescribes the position of a raw score in terms of its distance from the mean, measured in standard deviations:
𝑧=
𝑥−𝜇

𝜎
 The combined approach catches more anomalies than either method alone.
Key takeaway: Sometimes simple, well-chosen algorithms outperform complex ones, especially when you have limited data.
 

Fig 3: The anomaly detector flags unusual transactions, which stand out in the timeline | Image by Author

 


# Step 3: Designing Visualizations That Answer Questions
  Visualizations should answer questions, not just show data. I used Plotly for interactive charts because it allows users to explore the data themselves. Here are the design principles I followed:
	1.	Consistent color coding: Red for expenses, green for income
	2.	Context through comparison: Show income vs. expenses side by side
	3.	Progressive disclosure: Show a summary first, then let users drill down
For example, the spending breakdown uses a donut chart with a hole in the middle for a cleaner look:
import plotly.express as px

fig = px.pie(
    category_totals,
    values="Amount",
    names="Category",
    hole=0.4,
    color_discrete_map=CATEGORY_COLORS
)
 
Streamlit makes it easy to add these charts with st.plotly_chart() and build a responsive dashboard.
 

Fig 4: Multiple chart types give users different perspectives on the same data | Image by Author

 


# Step 4: Integrating a Local Large Language Model for Natural Language Insights
  The final piece was generating human-readable insights. I chose to integrate Ollama, a tool for running LLMs locally. Why local instead of calling OpenAI or Claude?
	1.	Privacy: Bank data never leaves the machine
	2.	Cost: Unlimited queries, zero API fees
	3.	Speed: No network latency (though generation still takes a few seconds)
 
// Streaming for Better User Experience
LLMs can take several seconds to generate a response. Streamlit shows tokens as they arrive, making the wait feel shorter. Here is a simple implementation using requests with streaming:
import requests
import json

def generate(self, prompt):
    response = requests.post(
        f"{self.base_url}/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": True},
        stream=True
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            yield data.get("response", "")
 
In Streamlit, you can display this with st.write_stream().
st.write_stream(llm.get_overall_insights(df))
 
// Prompt Engineering for Financial Data
The key to useful LLM output is a structured prompt that includes actual data. For example:
prompt = f"""Analyze this financial summary:
- Total Income: ${income:,.2f}
- Total Expenses: ${expenses:,.2f}
- Top Category: {top_category}
- Largest Anomaly: {anomaly_desc}

Provide 2-3 actionable recommendations based on this data."""
 
This gives the model concrete numbers to work with, leading to more relevant insights.
 

Fig 5: The upload interface is simple; choose a CSV and let the AI do the rest | Image by Author

 

// Running the Application
Getting started is straightforward. You will need Python installed, then run:
pip install -r requirements.txt

# Optional, for AI insights
ollama pull llama3.2

streamlit run app.py
 
Upload any bank CSV (the app auto-detects the format), and within seconds, you will see a dashboard with categorized transactions, anomalies, and AI-generated insights.
 

# Conclusion
  This project taught me that building something functional is just the beginning. The real learning happened when I asked why each piece works:
	•	Why auto-detect columns? Because real-world data does not follow your schema. Building a flexible pipeline saves hours of manual cleanup.
	•	Why Isolation Forest? Because small datasets need algorithms designed for them. You do not always need deep learning.
	•	Why local LLMs? Because privacy and cost matter in production. Running models locally is now practical and powerful.
These lessons apply far beyond personal finance, whether you are analyzing sales data, server logs, or scientific measurements. The same principles of robust preprocessing, pragmatic modeling, and privacy-aware AI will serve you in any data project.
The complete source code is available on GitHub. Fork it, extend it, and make it your own. If you build something cool with it, I would love to hear about it.
 

// References
	•	Streamlit Documentation — Framework for building data apps
	•	scikit-learn: Isolation Forest — Anomaly detection algorithm
	•	Ollama — Run large language models locally
	•	Plotly Python — Interactive visualization library
	•	Pandas Documentation — Data manipulation in Python
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/3c8d6a6d-70aa-45fc-870d-a793bb8257a6" />

