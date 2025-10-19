import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Paths
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
train_path = os.path.join(data_dir, "train_clean.csv")
test_path = os.path.join(data_dir, "test_clean.csv")
model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(model_dir, exist_ok=True)

# Load cleaned data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.transform(test_df["clean_text"])

y_train = train_df["label"]
y_test = test_df["label"]

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, os.path.join(model_dir, "text_classifier.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
print(f"\n💾 Model and vectorizer saved in: {model_dir}")
