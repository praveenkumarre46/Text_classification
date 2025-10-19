import pandas as pd
import os
import requests
from io import StringIO

# Create data folder path
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

# We'll use a small open dataset hosted by Kaggle mirror for demo
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"

print("📥 Downloading dataset...")
response = requests.get(url)
if response.status_code != 200:
    raise Exception("❌ Failed to download dataset!")

# Read CSV data into pandas
df = pd.read_csv(StringIO(response.text))

# Rename and map to our fake complaint structure
df = df.rename(columns={"tweet": "consumer_complaint_narrative", "label": "label"})

# Keep only 1000 samples for quick testing
df = df.sample(1000, random_state=42).reset_index(drop=True)

# Save cleaned sample
output_path = os.path.join(data_dir, "consumer_complaints_sample.csv")
df.to_csv(output_path, index=False)

print(f"✅ Dataset downloaded and saved to: {output_path}")
print(df.head())
