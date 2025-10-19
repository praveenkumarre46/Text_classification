import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# Paths
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
input_path = os.path.join(data_dir, "consumer_complaints_sample.csv")
output_train = os.path.join(data_dir, "train_clean.csv")
output_test = os.path.join(data_dir, "test_clean.csv")

# Load data
df = pd.read_csv(input_path)
print(f"Loaded {len(df)} records")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#\w+", "", text)     # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text) # keep letters and spaces only
    text = re.sub(r"\s+", " ", text).strip()  # clean spaces
    return text

# Apply cleaning
df["clean_text"] = df["consumer_complaint_narrative"].apply(clean_text)

# Drop rows with empty text
df = df[df["clean_text"].str.strip() != ""]

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save cleaned datasets
train_df.to_csv(output_train, index=False)
test_df.to_csv(output_test, index=False)

print(f"✅ Cleaned and split data saved to:")
print(f"   {output_train}")
print(f"   {output_test}")
print("\nSample cleaned text:")
print(train_df.head(5)[["clean_text", "label"]])
