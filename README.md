# Customer_Insurance_Prediction

This project aims to predict whether a customer will purchase an insurance policy based on demographic features using various classification algorithms. The implementation includes data preprocessing, model training, evaluation, and performance comparison.
---
## Project Structure
Customer_Insurance_Prediction/
├── data/ # Dataset (not uploaded to GitHub)
├── plots/ # Confusion matrices and model comparison plot
├── report/ # Final project report (PDF)
├── src/
│ ├── preprocess.py # Data loading and preprocessing
│ ├── train_models.py # Training classification models
│ ├── evaluate_models.py # Model evaluation and visualization
│ └── utils.py # Utility functions
├── Customer_Insurance_Prediction.ipynb
├── main.py # Main script to run the pipeline
├── requirements.txt # Required Python packages
├── README.md # Project documentation
└── .gitignore # Files/directories to exclude from Git
---
## Dataset
- **File**: `Social_Network_Ads.csv`
- **Features**:
  - Age
  - Estimated Salary
- **Target**:
  - Purchased (0 = No, 1 = Yes)
---
## Models Used
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree Classifier  
- Random Forest Classifier  
---
## Workflow
1. **Preprocessing**: Scaling and train-test split  
2. **Training**: Fit five classifiers  
3. **Evaluation**: Accuracy, confusion matrix, classification report  
4. **Visualization**: Save confusion matrices and accuracy comparison plot
---
## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Place Social_Network_Ads.csv in the data/ folder.
3. Run the complete pipeline: python main.py
---
##Results
Confusion matrix images and a bar chart comparing model accuracies are saved in the plots/ directory.
---
##License
For educational and academic use only. Please give credit if reused.
