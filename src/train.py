import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from preprocess import clean_text

DATA_PATH = "data/support_emails.csv"
MODEL_PATH = "email_classifier.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Combine subject + body as input text
df["text"] = (df["subject"].astype(str) + " " + df["body"].astype(str)).apply(clean_text)

# Labeling function based on subject keywords
def label_email(subject):
    subject = str(subject).lower()
    if "billing" in subject or "payment" in subject or "charged" in subject:
        return "Billing"
    elif "login" in subject or "account" in subject or "access" in subject or "password" in subject:
        return "Login/Account"
    elif "api" in subject or "integration" in subject:
        return "API/Integration"
    else:
        return "General"

df["label"] = df["subject"].apply(label_email)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# Build pipeline (TF-IDF + Logistic Regression)
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=2000))
])

# Train
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")
