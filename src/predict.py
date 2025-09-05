import joblib
import argparse
from preprocess import clean_text

MODEL_PATH = "email_classifier.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict(subject: str, body: str) -> str:
    text = clean_text(f"{subject} {body}")
    model = load_model()
    return model.predict([text])[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict support email category")
    parser.add_argument("--subject", type=str, required=True, help="Email subject")
    parser.add_argument("--body", type=str, required=True, help="Email body")
    args = parser.parse_args()

    label = predict(args.subject, args.body)
    print(f"📧 Predicted Category: {label}")
