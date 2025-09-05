# AI Engineer Fresher Challenge — Email Classification

## 📌 Problem
Classify customer support emails into categories so an agent can triage faster.

**Target categories:**
- Billing
- Login/Account
- API/Integration
- General

## 📂 Structure
```
AI-Engineer-Challenge/
├── data/
│   └── support_emails.csv
├── notebooks/
│   └── email_classification.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## ⚙️ Setup
```bash
pip install -r requirements.txt
```

## 🚀 Train
```bash
python src/train.py
```

## 🔮 Predict
```bash
python src/predict.py --subject "Payment charged twice" --body "I was billed two times for August"
```

## 📝 Notes
- Labels are derived from subject keywords as a simple baseline.
- Model: TF-IDF + Logistic Regression.
- You can extend with better labeling, more data, or transformer embeddings later.
