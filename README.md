# Customer-Insurance-Prediction

This project focuses on predicting whether a customer will purchase insurance based on demographic data using multiple classification algorithms. The models are trained and compared to identify the most accurate approach for binary classification.

---

## Project Structure

```
Customer-Insurance-Prediction/
├── data/                            # Input CSV file (excluded from GitHub)
├── plots/                           # Confusion matrices, comparison charts
├── report/                          # Final report in PDF
├── src/
│   ├── preprocess.py                # Data loading and scaling
│   ├── train_models.py              # Model training functions
│   ├── evaluate_models.py           # Confusion matrix and accuracy evaluation
│   └── utils.py                     # Optional visualization helpers
├── Customer_Insurance_Prediction.ipynb
├── main.py                          # Pipeline runner script
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Dataset

- **File**: `Social_Network_Ads.csv`
- **Features**: Age, Estimated Salary
- **Target**: Purchased (0 or 1)

---

##  Models Trained

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier

---

##  Evaluation

Each model is evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

Visual results are saved under `plots/`.

---

##  How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place `Social_Network_Ads.csv` in the `data/` folder.

3. Run:
   ```bash
   python main.py
   ```

---

##  Report

The full report is available at:
```
report/Customer_Insurance_Report.pdf
```

---

## Author

- Vadan Datta
