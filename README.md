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

1. Clone or download the repository.
2. Open terminal in project root.
3. Create a virtual environment:

**Windows PowerShell**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

Linux/macOS

'''python3 -m venv venv
source venv/bin/activate'''

Install dependencies:

'''pip install --upgrade pip
pip install -r requirements.txt'''

Running the Project
1. Preprocess Data

'''python src/preprocess.py'''
Cleans the text

Splits into train/test

Saves train_clean.csv and test_clean.csv in data/
Screenshot:
![Preprocessing](screenshots/Screenshot_2025-10-19_153015.png)
2. Train Model

'''python src/train_model.py'''

Trains Multinomial Naive Bayes

Evaluates accuracy and classification report

Saves model and vectorizer in models/

Screenshot:
![Training Model](screenshots/Screenshot_2025-10-19_153059.png)
3. Make Predictions

'''python src/predict.py'''
Enter new text in terminal

Outputs predicted category

Screenshot:
![Make Prediction](screenshots/Screenshot_2025-10-19_153144.png)
Notes

Dataset used is a sample dataset for demonstration purposes.

Accuracy may vary due to small sample size.

Preprocessing removes URLs, mentions, hashtags, and non-alphabetic characters.

Ensure system date/time and your username are visible in screenshots.

Author

Pappireddy Praveen Kumar Reddy
