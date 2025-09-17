# Credit Card Fraud Detection

This project demonstrates a machine learning workflow for detecting fraudulent credit card transactions using Python and scikit-learn. It covers data preprocessing, handling class imbalance, model training, evaluation, and visualization.

## Dataset

- `creditcard.csv`: Contains anonymized credit card transactions labeled as fraudulent or legitimate.

## Workflow

1. **Data Import & Exploration**
   - Load data with pandas.
   - Explore class distribution and feature statistics.
   - Visualize data and check for missing/duplicate values.

2. **Preprocessing**
   - Remove duplicates.
   - Standardize the `Amount` feature.
   - Drop the `Time` column.

3. **Model Training (Imbalanced Data)**
   - Split data into train/test sets.
   - Train Logistic Regression, Random Forest, and Decision Tree classifiers.
   - Evaluate using classification reports.

4. **Handling Imbalanced Data**
   - Apply SMOTE oversampling to balance classes.
   - Retrain models and compare performance.

5. **Model Selection & Evaluation**
   - Select Random Forest for final evaluation.
   - Visualize confusion matrix and ROC curve.
   - Calculate Gini coefficient.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

## Usage

1. Clone the repository and place `creditcard.csv` in the project folder.
2. Open `Credit_card_fraud_detection.ipynb` in Jupyter Notebook or VS Code.
3. Run the notebook cells sequentially to reproduce the analysis.

## Results

- Random Forest provides robust performance for fraud detection.
- ROC and Gini metrics indicate strong model discrimination.

## References

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- scikit-learn documentation
