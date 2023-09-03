"""
Name: Mohammad Al Dridi
ID: 400429815
Purpose: This code will analyze a dataset that contains 20 attributes. These attributes will be used whether a
credit card applicant in Germany will be accepted or rejected. We have 1000 instances recorded.
We will compare two ML modules based on time needed for training and results of confusion matrix
"""
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


def main():
    # import data from file and parse
    df = pd.read_csv('german.data',
                     header=None,
                     sep=' '
                     )

    # Format data to X and y -- x: attributes to be used in classification(independent) y: target variable(dependent)
    X = df.drop(df.columns[-1], axis=1).copy()
    y = df[df.columns[-1]].copy()

    # Change categorical data to binary data using one-hot encoding
    columns_to_encode = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    X_encoded = pd.get_dummies(X, columns=columns_to_encode)

    # Split data to training and test set, 75% and 25% respectively
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=30)

    # We will make sure that the data is mean centered and scaled to a SD of 1 due to the usage of SVM
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)

    # Use cross validation to estimate the best parameters for SVM
    param_grid = [
        {'C': [0.5, 1, 10, 100],
         'gamma': ['scale', 1, 0.1, 0.001, 0.0001]
         }
    ]
    best_params = GridSearchCV(SVC(), param_grid=param_grid, cv=10)
    best_params.fit(X_train_scaled, y_train)

    # Use SVM classifier and train it based on best parameters found and default kernel rbf
    classifier_svm = SVC(random_state=30, C=best_params.best_params_['C'], gamma=best_params.best_params_['gamma'])
    # Measure computational time for training
    starting_time_training_svm = time.time()
    classifier_svm.fit(X_train_scaled, y_train)
    end_time_training_svm = time.time()
    print(f"Time needed to train by SVM: {end_time_training_svm - starting_time_training_svm} seconds")

    # use trained data on test data and measure computational time
    starting_time_clf_svm = time.time()
    classifier_svm_predictions = classifier_svm.predict(X_test_scaled)
    end_time_clf_svm = time.time()
    print(f"Time needed to classify by SVM: {end_time_clf_svm - starting_time_clf_svm} seconds")

    # Create a confusion matrix
    cm_svm = confusion_matrix(y_test, classifier_svm_predictions)

    # Print the confusion matrix
    disp_cm_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=[True, False])
    disp_cm_svm.plot()

    # Train a Fischer Linear  Analysis model on the split data set
    classifier_LDA = LinearDiscriminantAnalysis()
    # Measure computational time need to train LDA
    starting_time_training_LDA = time.time()
    classifier_LDA.fit(X_train_scaled, y_train)
    end_time_training_LDA = time.time()
    print(f"Time needed to train by LDA: {end_time_training_LDA - starting_time_training_LDA} seconds")

    # Calculate predictions and computational time needed for classification
    starting_time_clf_LDA = time.time()
    classifier_LDA_predictions = classifier_LDA.predict(X_test_scaled)
    end_time_clf_LDA = time.time()
    print(f"Time needed to classify by LDA: {end_time_clf_LDA - starting_time_clf_LDA} seconds")

    # create a confusion matrix and display it
    cm_LDA = confusion_matrix(y_test, classifier_LDA_predictions)

    disp_cm_LDA = ConfusionMatrixDisplay(confusion_matrix=cm_LDA, display_labels=[True, False])
    disp_cm_LDA.plot()
    plt.show()


main()
