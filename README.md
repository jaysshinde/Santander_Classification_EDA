# Santander Customer Satisfaction Prediction

This repository contains the code and results for predicting customer satisfaction using the Santander Customer Satisfaction dataset. The main goal of this project is to get started with machine learning using Python's Scikit Learn library, understand exploratory data analysis (EDA), and construct decision tree classification models.

## Dataset

The dataset used in this project is from a concluded Kaggle competition with a prize of $60,000. It is about predicting customer satisfaction for Santander Bank customers. You can download the dataset from the following links:

- Training set: [Santander Customer Satisfaction - TRAIN.csv](https://www.kaggle.com/c/santander-customer-satisfaction/data?select=train.csv)
- Test set: [Santander Customer Satisfaction - TEST-Without TARGET.csv](https://www.kaggle.com/c/santander-customer-satisfaction/data?select=test.csv)

## Requirements

- Python 3.6 or higher
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn

## Steps

1. **Exploratory Data Analysis (EDA):** We'll begin by performing EDA on the dataset to understand its major characteristics, including heatmaps for correlation and handling under-representation using undersampling.
2. **Feature Selection:** We'll remove highly correlated attributes to improve model performance.
3. **Decision Tree Classifier:** We'll train and validate a decision tree classification model using Scikit-learn and explore different hyperparameters.
4. **Report and Presentation:** We'll create a report discussing the dataset, problem, and machine learning models. 
## Instructions

1. Clone this repository and navigate to the project folder.
2. Install the required packages (if not already installed).
3. Run the Jupyter Notebook or Python script containing the EDA, decision tree classifier, and parameter tuning.


## Results

Printing the precision and recall, among other metrics:

              precision    recall  f1-score   support

           0       0.98      0.79      0.88     14602
           1       0.12      0.70      0.21       602
    accuracy                           0.79     15204
   macro avg       0.55      0.74      0.54     15204
weighted avg       0.95      0.79      0.85     15204


