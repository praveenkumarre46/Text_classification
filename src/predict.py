import joblib
import os

# Paths
base_dir = os.path.dirname(__file__)
model_dir = os.path.join(base_dir, "..", "models")

model_path = os.path.join(model_dir, "text_classifier.pkl")
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_text(text):
    cleaned = text.lower()
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    label_map = {0: "Negative / Other", 1: "Positive / Complaint"}
    return label_map.get(pred, "Unknown")

if __name__ == "__main__":
    print("✅ Model loaded. Enter text to classify (type 'exit' to quit):\n")
    while True:
        text = input("Enter complaint text: ")
        if text.lower() == "exit":
            break
        prediction = predict_text(text)
        print(f"🔹 Prediction: {prediction}\n")
