# from train.train_data import FraudDetectionPipeline
import logging
import os
import sys

from load.load_data import DataRetriever, RetrieveURLZIP_ExtractFile
from preprocess.preprocess_data import Change_TransactionType

# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from utilities.logging import MyLogger

# root folder

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)


# ZIP URL
# IT's a big file!! ZIP (180Mb) CSV(480Mb)
# Instead of trying to have a direct link in Kaggle, we choose to upload the file on Google Drive....
# Nevertheless..we had to create an Google Drive key to create a direct link to the file.. otherwise there is button
#    to manually download files bigger than 150Mb
ZIPURL = "https://www.googleapis.com/drive/v3/files/1_rd4Jy9bbpjCyl92zqBor5bKQgD00Y0L?alt=media&key=AIzaSyCUt59fyn0PV3Ar7HjnyW2_r6FBe6AUyrM"  # ZIP Url
CSVFILE = "PS_20174392719_1491204439457_log.csv"

# refactored folder
REFACTORED_DIRECTORY = ""
DATASETS_DIR = "./data/"  # Directory where data will be unzip.
RETRIEVED_DATA = (
    "retrieved_data.csv"  # File name for the retrieved data without irrelevant columns
)


class Retrieve_Files:
    def __init__(self):
        self = self
        self.logfile = MyLogger("Retrieve_Files", logging.DEBUG, __name__)

    def retrieve_files(self):
        Flag_result = "Ok"
        # Change location to the refactored directory
        # print(os.getcwd())
        # os.chdir(REFACTORED_DIRECTORY)
        # print(os.getcwd())

        # Check if we already have the file.
        # It's a big file so try not to downloaded and unzip it every run
        if not os.path.isfile(DATASETS_DIR + CSVFILE):
            # if we don't have the CSV, proceed to download it and unzip it

            self.logfile.info(
                f"Please wait downloading ZIP File '{CSVFILE}' (190Mb ZIP - 480Mb CSV"
            )
            data_retriever = RetrieveURLZIP_ExtractFile(ZIPURL)
            result = data_retriever.retrieve_data()
            # print(result)

        # if we still don't have the CSV file: Houston we have a problem!
        if not os.path.isfile(DATASETS_DIR + CSVFILE):
            self.logfile.critical(
                f"There was a problem downloading the ZIP File or unzipping the CSV '{DATASETS_DIR+CSVFILE}' "
            )
            Flag_result = "Error"
        else:
            self.logfile.info(f"File '{DATASETS_DIR+CSVFILE}' in place!")

        # if we already have the retrieved file, skip this section
        if not os.path.isfile(DATASETS_DIR + RETRIEVED_DATA):
            data_retrieve = DataRetriever(DATASETS_DIR + CSVFILE)
            result = data_retrieve.retrieve_data()
            self.logfile.info(result)
        else:
            result = f"File '{DATASETS_DIR+RETRIEVED_DATA}' in place!"
            self.logfile.info(result)

        if Flag_result == "Ok":
            self.logfile.info("Data retrieved")
            return "Data retrieved"
        else:
            self.logfile.error("FILES NOT AVAILABLE")
            return "FILES NOT AVAILABLE!"


class FraudDetectionPipeline:
    """
    A class representing the Online Fraud Detection data processing and modeling pipeline.

    Attributes:
        NUMERICAL_VARS (list): A list of numerical variables in the dataset.
        CATEGORICAL_VARS_WITH_NA (list): A list of categorical variables with missing values.
        NUMERICAL_VARS_WITH_NA (list): A list of numerical variables with missing values.
        CATEGORICAL_VARS (list): A list of categorical variables in the dataset.
        SEED_MODEL (int): A seed value for reproducibility.

    Methods:
        create_pipeline(): Create and return the Titanic data processing pipeline.
    """

    def __init__(
        self,
        seed_model,
        numerical_vars,
        categorical_vars_with_na,
        numerical_vars_with_na,
        categorical_vars,
        selected_features,
    ):
        self.SEED_MODEL = seed_model
        self.NUMERICAL_VARS = numerical_vars
        self.CATEGORICAL_VARS_WITH_NA = categorical_vars_with_na
        self.NUMERICAL_VARS_WITH_NA = numerical_vars_with_na
        self.CATEGORICAL_VARS = categorical_vars
        self.SEED_MODEL = seed_model
        self.SELECTED_FEATURES = selected_features
        # logging instance
        self.logfile = MyLogger("FraudDetectionPipeline", logging.DEBUG, __name__)

    def create_pipeline(self):
        """
        Create and return the DataFraud data processing pipeline.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """
        self.logfile.debug("Create data pipeline")
        self.PIPELINE = Pipeline(
            [
                ("Change transaction type", Change_TransactionType()),
            ]
        )
        return self.PIPELINE

    def fit_DecisionTree(self, X_train, y_train):
        """
        Fit a Decision Tree model using the predefined data preprocessing pipeline.

        Parameters:
        - X_train (pandas.DataFrame or numpy.ndarray): The training input data.
        - y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
        - Decision Tree _model (DecisionTree): The fitted Decision Tree model.
        """
        self.logfile.debug("Train Decision Model")
        Decision_tree_model = DecisionTreeClassifier()
        pipeline = self.create_pipeline()
        pipeline.fit(X_train.values, y_train)
        Decision_tree_model.fit(pipeline.transform(X_train).values, y_train)
        return Decision_tree_model

    def transform_test_data(self, X_test):
        """
        Apply the data preprocessing pipeline on the test data.

        Parameters:
        - X_test (pandas.DataFrame or numpy.ndarray): The test input data.

        Returns:
        - transformed_data (pandas.DataFrame or numpy.ndarray): The preprocessed test data.
        """
        self.logfile.debug("Transform raw data w/transformations needed in Model")
        pipeline = self.create_pipeline()
        return pipeline.transform(X_test)
