"""Main module."""
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from train.train_data import FraudDetectionPipeline, Retrieve_Files
from utilities.logging import MyLogger

# Define file locations

# ZIP URL
# IT's a big file!! ZIP (180Mb) CSV(480Mb)
# Instead of trying to have a direct link in Kaggle, we choose to upload the file on Google Drive....
# Nevertheless..we had to create an Google Drive key to create a direct link to the file.. otherwise there is button
#    to manually download files bigger than 150Mb
ZIPURL = "https://www.googleapis.com/drive/v3/files/1_rd4Jy9bbpjCyl92zqBor5bKQgD00Y0L?alt=media&key=AIzaSyCUt59fyn0PV3Ar7HjnyW2_r6FBe6AUyrM"  # ZIP Url


CSVFILE = "PS_20174392719_1491204439457_log.csv"

# refactored folder
# REFACTORED_DIRECTORY = "/Users/francisco.torres/Documents/GitHub/MLOps_proj2"
DATASETS_DIR = "./data/"  # Directory where data will be unzip.

RETRIEVED_DATA = (
    "retrieved_data.csv"  # File name for the retrieved data without irrelevant columns
)


# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(current_dir)

ROOT_DIRECTORY = parent_dir  # "/Users/francisco.torres/Documents/GitHub/MLOps_project/Refactor/mlops_project/mlops_project"
APP_DIRECTORY = current_dir

DATASETS_DIR = "./data/"  # Directory where data will be unzip.
RETRIEVED_DATA = (
    "retrieved_data.csv"  # File name for the retrieved data without irrelevant columns
)


TRAIN_DATA_FILE = DATASETS_DIR + "train.csv"
TEST_DATA_FILE = DATASETS_DIR + "test.csv"

TRAINED_MODEL_DIR = "./models/"
PIPELINE_NAME = "KNN"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output.pkl"

# Persist/Save model
SAVE_FILE_NAME = f"{PIPELINE_SAVE_FILE}"
SAVE_PATH = TRAINED_MODEL_DIR + SAVE_FILE_NAME


# Define attributes to work with

TARGET = "isFraud"
FEATURES = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]
NUMERICAL_VARS = ["amount", "oldbalanceOrg", "newbalanceOrig"]
CATEGORICAL_VARS = ["type"]
NUMERICAL_VARS_WITH_NA = []
CATEGORICAL_VARS_WITH_NA = []


SELECTED_FEATURES = [
    "type_CASH_OUT",
    "type_PAYMENT",
    "type_CASH_IN",
    "type_TRANSFER",
    "type_rare",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
]
SELECTED_FEATURES = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]
SEED_MODEL = 404


if __name__ == "__main__":
    # create logging file instance

    logfile = MyLogger("Main_TEST", logging.DEBUG, __name__)

    # Change location to the refactored directory
    # print(os.getcwd())
    logfile.info(f"Current directory : {os.getcwd()}")
    os.chdir(APP_DIRECTORY)
    print(os.getcwd())
    logfile.info(f"Refactored directory : {os.getcwd()}")

    # This class will retrieve ZIP file and extract csv
    retrieve_files = Retrieve_Files()
    result = retrieve_files.retrieve_files()
    # Instantiate the FraudDetectionPipeline class
    fraud_data_pipeline = FraudDetectionPipeline(
        seed_model=SEED_MODEL,
        numerical_vars=NUMERICAL_VARS,
        categorical_vars_with_na=CATEGORICAL_VARS_WITH_NA,
        numerical_vars_with_na=NUMERICAL_VARS_WITH_NA,
        categorical_vars=CATEGORICAL_VARS,
        selected_features=SELECTED_FEATURES,
    )

    # Read data
    logfile.debug(f"CSV file: {DATASETS_DIR + RETRIEVED_DATA}")
    df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)

    # Split data
    logfile.debug("Split dataset 80% Train / 20% Test")

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=3
    )

    # Fit the model
    logfile.debug("Model fitting KNN")
    KNNmodel = fraud_data_pipeline.fit_KNN(X_train, y_train)

    result = joblib.dump(KNNmodel, SAVE_PATH)
    logfile.info(f"Model saved in {result}")

    # calculate metrics: roc-auc / accuracy /confusion matrix / classification_report  over the Test dataset
    X_test = fraud_data_pipeline.PIPELINE.fit_transform(X_test)
    class_pred = KNNmodel.predict(X_test)
    proba_pred = KNNmodel.predict_proba(X_test)[:, 1]
    print(f"test roc-auc : {roc_auc_score(y_test, proba_pred)}")
    #  test roc-auc : 0.9297945977633371
    logfile.info(f"Testing model ROC-AUC:{roc_auc_score(y_test, proba_pred)}")

    print(f"test accuracy: {accuracy_score(y_test, class_pred)}")
    logfile.info(f"Testing model accuracy: {accuracy_score(y_test, class_pred)}")
    #  test accuracy: 0.9996730906450487

    print(confusion_matrix(y_test, class_pred))
    # [[1270686     184]
    # [    232    1422]]
    print(classification_report(y_test, class_pred))
    #              precision    recall  f1-score   support

    #           0       1.00      1.00      1.00   1270870
    #           1       0.89      0.86      0.87      1654

    #    accuracy                           1.00   1272524
    #   macro avg       0.94      0.93      0.94   1272524
    # weighted avg       1.00      1.00      1.00   1272524

    logfile.info("Confusion Matrix")
    logfile.info(confusion_matrix(y_test, class_pred))

    logfile.info("Classification Report")
    logfile.info(classification_report(y_test, class_pred))

    trained_model = joblib.load(filename=SAVE_PATH)

    logfile.debug("Test Trained model")
    logfile.debug("NO Fraud cases -- Expected [0's]")
    # NO FRAUD CASES (0)
    test_data = np.array(
        [
            [3, 196821.07, 1728770.3, 1925591.38],
            [1, 123974.95, 27160.24, 0],
            [1, 123974.95, 27160.24, 0],
            [3, 90646.09, 1638124.21, 1728770.3],
            [3, 87541.63, 1925591.38, 2013133.01],
            [3, 87541.63, 1925591.38, 2013133.01],
            [2, 2625.76, 29786, 27160.24],
            [2, 2625.76, 29786, 27160.24],
        ]
    ).astype(np.float32)
    logfile.debug(trained_model.predict(test_data))

    # FRAUD CASES (1)
    logfile.debug("Fraud cases -- Expected [1's]")
    test_data = np.array(
        [
            [4, 10000000, 12930418.44, 2930418.44],
            [1, 1277212.77, 1277212.77, 0],
            [4, 1277212.77, 1277212.77, 0],
            [1, 416001.33, 0, 0],
            [1, 235238.66, 235238.66, 0],
            [4, 235238.66, 235238.66, 0],
            [1, 132842.64, 4499.08, 0],
        ]
    ).astype(np.float32)
    logfile.debug(trained_model.predict(test_data))
