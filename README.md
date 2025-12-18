# Text Classification on Consumer Complaints

## Project Overview

This project builds a machine learning based text classification system to categorize consumer complaint texts into predefined categories.
It uses TF-IDF vectorization for text preprocessing and a Multinomial Naive Bayes classifier for prediction.
The goal is to automate classification of text data and demonstrate an end-to-end ML workflow.


---

## Project Structure

Text_classification/
├── data/ # Raw and cleaned dataset files
├── models/ # Saved model and vectorizer
├── screenshots/ # Results and visual explanations
├── src/ # Script files (preprocessing, training, prediction)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Machine Learning Approach

Text data is first cleaned to remove noise like URLs, hashtags, and non-alphabetic characters.
TF-IDF vectorization is applied to convert text into numerical features.
A Multinomial Naive Bayes classifier is trained on the transformed features to predict the text category.
The trained model and vectorizer are saved for reuse.

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/praveenkumarre46/Text_classification.git
cd Text_classification
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python src/preprocess.py
python src/train_model.py
python src/predict.py

