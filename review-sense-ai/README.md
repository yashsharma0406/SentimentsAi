# ReviewSense AI 🧠📊

[![CI](https://github.com/Gujjar-Pranav/review-sense-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Gujjar-Pranav/review-sense-ai/actions/workflows/ci.yml)

🔗 **Live App:** https://review-sense-ai-mvvd48vdsasmys7ecjenpa.streamlit.app/  
*(If sleeping, open once to wake it up)*

---

**ReviewSense AI** is a trust-aware review intelligence dashboard that transforms raw customer reviews into **clear insights, risks, and actions**.  
It combines machine learning, confidence scoring, and explainable analytics to help teams understand *what customers feel*, *where the model is uncertain*, and *what needs human attention*.

---

## 🚀 What This Project Does

ReviewSense AI analyzes customer reviews (Amazon-style) and provides:

- Sentiment classification (Positive / Negative / Mixed)
- Confidence & risk scoring for each prediction
- Identification of **tricky reviews** where AI struggles
- Executive-level insights for decision-makers
- A polished, interactive **Streamlit dashboard**

This project is designed to be both **ML-practical** and **business-ready**.

---

## 🧩 Key Features

### 🛡️ Trust & Confidence Dashboard
- Negative risk percentage
- Low-confidence review detection
- Auto-approve vs manual-review zones
- Clear operational recommendations

### 🧪 Tricky Reviews (AI Limitations)
Detects reviews that are hard for AI to judge, including:
- Mixed sentiment
- Negation (e.g. *"not bad"*)
- Confusing or vague wording
- Strong tone / emphasis (caps, punctuation)
- Uncertain or borderline predictions

### 📊 Business Insights (Executive View)
- Overall sentiment distribution
- Focus index (where to fix first)
- Top praise themes
- Example high-confidence praise & complaints
- Shareable plain-English summary

### 🔍 Drilldowns & Transparency
- Filter by confidence threshold
- Category-based analysis
- Download-ready results table
- Clear explanation of why reviews need human review

---

## 🧠 Machine Learning Pipeline

- Text preprocessing & TF-IDF feature extraction
- Calibrated sentiment modeling
- Probability-based confidence scoring
- Error analysis and misclassification reports
- Model comparison utilities

---

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** – interactive dashboard
- **scikit-learn** – ML models & calibration
- **pandas / numpy** – data processing
- **Plotly** – rich visualizations
- **Joblib** – model persistence

---

## Screenshots 

<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 50 15" src="https://github.com/user-attachments/assets/880eea6c-8125-45ac-981d-1b68d9d72dec" />
<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 50 40" src="https://github.com/user-attachments/assets/0b5b888b-bd0a-4947-86a9-ff7b2cdf156b" />
<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 50 48" src="https://github.com/user-attachments/assets/a9f2f4db-cff5-4466-81fa-3b9b1c136913" />
<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 50 57" src="https://github.com/user-attachments/assets/3bbea809-f90f-4b4e-a556-ec7c68848077" />
<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 51 08" src="https://github.com/user-attachments/assets/88ade415-072d-46d7-a371-ab256fd58927" />
<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 51 18" src="https://github.com/user-attachments/assets/9e031a87-f2a9-47e0-9539-293ee78137ac" />
<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 51 32" src="https://github.com/user-attachments/assets/ff72ddc0-b7fa-42df-a646-1e56c084df5d" />
<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 51 46" src="https://github.com/user-attachments/assets/b58c13ee-16b2-4383-a4ab-6324784cb0b4" />
<img width="300" height="300" alt="Screenshot 2025-12-31 at 17 51 59" src="https://github.com/user-attachments/assets/cc661382-cacf-4e1f-8246-823895943601" />


## 📂 Project Structure

```text
review-sense-ai/
│
├── app/                    # Streamlit UI
│   ├── streamlit_app.py
│   ├── ui_helpers.py
│   └── visualizations.py
│
├── src/                    # ML & analysis pipeline
│   ├── preprocess.py
│   ├── calibrate_train.py
│   ├── modeling_compare.py
│   ├── error_analysis.py
│   ├── eda.py
│   ├── data_load.py
│   ├── config.py
│   └── utils.py
│
├── artifacts/              # Trained model (committed)
│   └── best_model_calibrated.joblib
│
├── outputs/reports/        # Evaluation artifacts
│   ├── misclassified.csv
│   ├── model_comparison.csv
│   └── calibrated_metrics.json
│
├── data/
│   └── amazonreviews.tsv
│
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md

▶️ How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/Gujjar-Pranav/review-sense-ai.git
cd review-sense-ai

2️⃣ Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ (Optional) Regenerate model & reports
python main.py

5️⃣ Run the Streamlit app
streamlit run app/streamlit_app.py

📈 Example Use Cases

Product teams prioritizing customer pain points

Analysts auditing ML confidence and failure modes

Businesses deciding when AI decisions need human review

Portfolio demonstration of Responsible AI design

🔒 Responsible AI Focus

ReviewSense AI explicitly highlights:

Where the model is uncertain

Why human review is required

How to safely operationalize ML predictions

This makes it suitable for real-world, high-stakes ML use cases.

🧪 CI/CD & Quality Gates

This repository includes production-grade CI/CD:

✅ GitHub Actions CI

✅ Ruff linting (PEP8 + modern Python)

✅ Python compile checks

✅ Artifact presence validation

✅ Fail-fast safety checks

All commits to main must pass CI before merging.

📌 Future Improvements

Model monitoring over time

Topic modeling for complaints

Multi-language support

A/B evaluation dashboard

Drift & confidence alerts

👤 Author

Pranav Gujjar
Machine Learning & Data Science

