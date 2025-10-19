# Task 5: Text Classification on Consumer Complaints

## Project Description
This project classifies consumer complaint texts into categories using **TF-IDF vectorization** and **Multinomial Naive Bayes**.

- Cleaned and preprocessed text data
- Trained a Naive Bayes model
- Implemented prediction script for new text inputs
- Demonstrated results with accuracy and classification report

---

## Project Structure

task5_text_classification/
│
├── data/ # Dataset files
├── models/ # Saved model and vectorizer
├── screenshots/ # Screenshots showing your work
├── src/ # Python scripts
├── README.md
└── requirements.txt


---

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