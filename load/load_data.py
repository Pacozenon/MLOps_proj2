import logging
import os
import re
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import sklearn

from utilities.logging import MyLogger


class RetrieveURLZIP_ExtractFile:
    """
    A class for retrieving a ZIP file from a given URL, UNZIP it on a ./Data/ directory for further analysis.
    Result

    Parameters:
        url (str): The URL from which the data will be loaded.
    Attributes:
        url (str): The URL from which the data will be loaded.

    Example usage:
    ```
    URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
    data_retriever = RetrieveURLZIP_ExtractFile(URL)
    result = data_retriever.retrieve_data()
    print(result)
    ```
    """

    DATASETS_DIR = "./data/"  # Directory where data will be saved.
    RETRIEVED_DATA = "retrieved_data.csv"  # File name for the retrieved data.

    def __init__(self, url):
        self.url = url
        self.logfile = MyLogger("RetrieveURLZIP_ExtractFile", logging.DEBUG, __name__)

    def retrieve_data(self):
        """
        Retrieves data from the specified URL, processes it, and stores it in a CSV file.

        Returns:
            str: A message indicating the location of the stored data.
        """

        DATASETS_DIR = "./data/"  # Directory where data will be unzip.

        # Create directory if it does not exist
        if not os.path.exists(DATASETS_DIR):
            os.makedirs(DATASETS_DIR)
            #    print(f"Directory '{DATASETS_DIR}' created successfully.")
            self.logfile.debug(f"Directory '{DATASETS_DIR}' created successfully.")
        else:
            self.logfile.debug(f"Directory '{DATASETS_DIR}' already exists.")

        # currentdir= os.curdir()

        # Retrieve zip file from specific URL
        # Unzip file to DATASET_DIR directory

        with urlopen(self.url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(DATASETS_DIR)
        ret = f"Data unzipped in {self.DATASETS_DIR}"
        self.logfile.info(ret)
        return ret


# Usage Example:
# URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
# data_retriever = DataRetriever(URL)
# result = data_retriever.retrieve_data()
# print(result)


class DataRetriever:
    """
    A class for retrieving data from a given FILE and processing it for further analysis.

    Parameters:
        csv_filename (str): The path+filename from which the data will be loaded.

    Attributes:
        csv_filename (str): The path+filename from which the data will be loaded.

    Example usage:
    ```
    CSV_FILENAME = './data/PS_log.csv'
    data_retriever = DataRetriever(CSV_FILENAME)
    result = data_retriever.retrieve_data()
    print(result)
    ```
    """

    DROP_COLS = [
        "step",
        "nameOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ]

    DATASETS_DIR = "./data/"  # Directory where data will be saved.
    RETRIEVED_DATA = "retrieved_data.csv"  # File name for the retrieved data.

    def __init__(self, csvfile):
        self.csvfile = csvfile

    def retrieve_data(self):
        """
        Retrieves data from the specified URL, processes it, and stores it in a CSV file.

        Returns:
            str: A message indicating the location of the stored data.
        """
        # Loading data from specific URL
        data = pd.read_csv(self.csvfile)

        # Drop irrelevant columns
        data.drop(self.DROP_COLS, axis=1, inplace=True)

        # Transform categorical attribute 'type'
        # data["type"] = data["type"].map(
        #    {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
        # )

        # Create directory if it does not exist
        if not os.path.exists(self.DATASETS_DIR):
            os.makedirs(self.DATASETS_DIR)
            print(f"Directory '{self.DATASETS_DIR}' created successfully.")
        else:
            print(f"Directory '{self.DATASETS_DIR}' already exists.")

        # Save data to CSV file
        data.to_csv(self.DATASETS_DIR + self.RETRIEVED_DATA, index=False)

        return f"Data stored in {self.DATASETS_DIR + self.RETRIEVED_DATA}"


# Usage Example:
# URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
# data_retriever = DataRetriever(URL)
# result = data_retriever.retrieve_data()
# print(result)
