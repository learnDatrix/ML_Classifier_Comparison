# ML_Classifier_Comparison
Comparing Two Classifiers - Fischer's Linear Discriminant Analysis &amp; Support Vector Machines

## Executive Summary

This README document provides an overview of a Python code that analyzes a dataset containing 20 attributes. The primary objective of this analysis is to determine whether a credit card applicant in Germany will be accepted or rejected based on these attributes. The dataset comprises 1000 instances, and we will compare the performance of two machine learning classifiers: Fischer's Linear Discriminant Analysis and Support Vector Machines. The evaluation criteria include the time required for training each model and the resulting confusion matrix.

## Briefing

### Dataset Source

The dataset used for this analysis was obtained from the UCI Machine Learning Repository (reference link: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)). This dataset contains information about credit card applicants in Germany.

### Dataset Attributes

The dataset consists of 20 attributes, which can be categorized as follows:

- 17 categorical attributes: These attributes represent categorical data such as employment status, housing type, and credit history.
- 3 continuous attributes: These attributes represent numerical data, such as age, credit amount, and duration of credit.

### Analysis Steps

1. **Data Preprocessing**: The code includes data preprocessing steps, such as handling missing values, one-hot encoding categorical variables, and scaling numerical features if necessary.

2. **Model Selection**:
   - Fischer's Linear Discriminant Analysis (LDA)
   - Support Vector Machines (SVM)

3. **Model Training and Evaluation**:
   - The code trains both LDA and SVM models on the preprocessed dataset.
   - The training time for each model is recorded.
   - The code generates confusion matrices to evaluate the classification performance of each model.

4. **Results**:
   - The results include the training time for both LDA and SVM models.
   - The confusion matrices provide insights into the model's ability to correctly classify credit card applicants as accepted or rejected.
  
   - Confusion Matrix of SVM:
      ![cf of SVM](/screenshots/SVM_cf)
     
   - Confusion Matrix of LDA:
      ![cf of LDA](/screenshots/LDA_cf)
## Usage

To run the code and perform the analysis:

1. Ensure you have Python installed on your system.

2. Install the necessary Python libraries and dependencies, such as NumPy, pandas, scikit-learn, and matplotlib.

