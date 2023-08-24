import logging
import os
import sys

import joblib
import pandas as pd
from classifier.classifier import ModelClassifier
from fastapi import FastAPI
from models.models import OnlineTX
from sklearn.model_selection import train_test_split
from starlette.responses import JSONResponse
from train.train_data import FraudDetectionPipeline, Retrieve_Files
from utilities.logging import MyLogger

# root folder

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

SEED_MODEL = 725

PIPELINE_NAME = "DecisionTree"

MODEL_DIRECTORY = f"{APP_DIRECTORY}\\models\\"  # "/Users/francisco.torres/Documents/GitHub/MLOps_project/Refactor/mlops_project/mlops_project/models/"
# TRAINED_MODEL_DIR = "./models/"
PIPELINE_NAME = "DecisionTree"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output.pkl"

# Persist/Save model
# SAVE_FILE_NAME = f"{PIPELINE_SAVE_FILE}"
MODEL_PATH = MODEL_DIRECTORY + PIPELINE_SAVE_FILE


logger = MyLogger("API Controller", logging.DEBUG, __name__)

app = FastAPI()


@app.get("/", status_code=200)
async def healthcheck():
    logger.info("ACTION->Online Fraud Classifier is all ready to go!")
    return "Online Fraud Classifier is all ready to go!"


@app.post("/predict")
def predict(Online_TX_features: OnlineTX) -> JSONResponse:
    # if we still don't have the CSV file: Houston we have a problem!
    if not os.path.isfile(MODEL_PATH):
        modeloutput = f"Compiled model NOT FOUND, please use the appropiate endpoint to regenerate the model '{MODEL_PATH}' "
        logger.critical(modeloutput)
    else:
        predictor = ModelClassifier(MODEL_PATH)
        X = [
            Online_TX_features.type,
            Online_TX_features.amount,
            Online_TX_features.oldbalanceOrg,
            Online_TX_features.newbalanceOrg,
        ]
        # print(f"Input values: {[X]}")
        # logger.info(f"Input values: {[X]}")
        # prediction = predictor.predict([X])
        # logger.info(f"Model Result: {prediction}")
        prediction = predictor.predict([X])
        logger.debug(
            f"Classification-> INPUT [\n"
            f"                         type: {Online_TX_features.type}\n"
            + f"                       amount: {Online_TX_features.amount} \n"
            + f"                oldbalanceorg: {Online_TX_features.oldbalanceOrg} \n"
            + f"                newbalanceorg: {Online_TX_features.newbalanceOrg}]"
            + f"\n                     Result-> {prediction}"
        )
        modeloutput = f"Resultado predicción: {prediction}"
    return JSONResponse(modeloutput)


@app.get("/train_model", status_code=200)
def train_model():
    # Change location to the root directory
    os.chdir(ROOT_DIRECTORY)

    logger.debug(f"ACTION -> Train model ")

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
    df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=3
    )

    # Fit the model

    DecisionTreeModel = fraud_data_pipeline.fit_DecisionTree(X_train, y_train)

    result = joblib.dump(DecisionTreeModel, MODEL_PATH)
    logger.debug(f"ACTION -> Train model saved in {result}")

    return "Trained model ready to go!"
