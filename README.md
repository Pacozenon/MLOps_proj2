___
### ITESM Instituto Tecnológico de Estudios Superiores de Monterrey
### Course:     MLOps Machine Learning Operations
#### Teacher:   Carlos Mejia
#### Student:   Francisco Javier Torres Zenón  A01688757
____

## References
* Dataset and baseline notebook copied from [Online Payments Fraud Detection Dataset | Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset) 
* Dataset: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset/download?datasetVersionNumber=1
* Baseline notebook: https://www.kaggle.com/code/nehahatti/online-payments-fraud-detection-project/notebook


# About this notebook
This notebook was taken and changed from [Kaggle](http://www.kaggle.com)

1. The task is to predict online payment fraud, given a number of features from online transfer/deposits transactions.

2. On Kaggle there were several notebooks related to this dataset(Decision Tree, Logistic Regresion, KNN, Gradient Boosting Classifier)
3. As a Baseline I choose one of the most accurated and simpler one, a notebook using the Decision Tree algorithm.

**Baseline Metrics**
```
Confussion Matrix
[[1270721     149]
 [     86    1568]]
```

```
 Classification Report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00   1270870
           1       0.91      0.95      0.93      1654

    accuracy                           1.00   1272524
   macro avg       0.96      0.97      0.97   1272524
weighted avg       1.00      1.00      1.00   1272524
```


## Scope

* Project focused on MLOps, where the key concepts of ML frameworks learned on this course were applied in a holistic approach.

* In this project we will apply the best practices in MLOPs to a baseline notebook to create a model to predict online payment fraud, ready to use via API.

### Out of Scope

* Since we have already have on the baseline a good recall(0.95) and F1-Score(0.93) metrics over the FRAUD cases, we will note explore another methods.

* Also we will not make an intensive feature analysis nor feature engineering.


## Online Payments Fraud Detection
### Introduction
The introduction of online payment systems has helped a lot in the ease of payments. But, at the same time, it increased in payment frauds. Online payment frauds can happen with anyone using any payment system, especially while making payments using a credit card. 

That is why detecting online payment fraud is very important for credit card companies to ensure that the customers are not getting charged for the products and services they never paid. 

Part I 
    Definition
    Scope
    Baseline

Part II 
    Virtual environments
    Unit tests
    Pre-commits
    Refactoring
    Lining and formatting
    Directory structure
    OOP (Classes, methods, transformers, pipelines)
    REST API - FastAPI

This session talks about one of the most important practices to be able to climb an ML system: refactorization. Topics such as the directories structure of an ML system are included, the weaknesses that a notebook has to use in production, and a demo to refactorize an existing project.

## Setup
### Virtual environment

1. Create a virtual environment with `Python 3.10+` from the root folder
    * Create venv
        ```bash
        python3.10 -m venv venv
        ```

    * Activate the virtual environment
        ```
        Linux:
              source venv/bin/activate
        Windows:
              ./venv/scripts/activate.ps1
        ```
2. Make sure you are on the root folder

        Windows
        ./mlops_proj2/
       
## Install all requerimients files

* General, API & PyTest packages 

        python -m pip install -r requirements-310.txt

## Activate pre-commit hooks

### Pre-commits
Pre-commits are automated checks that run on your code before you commit changes, helping ensure code quality and consistency. In this guide, we'll use the `pre-commit` tool to set up pre-commits for Python projects in Visual Studio Code (VSC).

### Prerequisites

1. Python is installed on your system.
2. Visual Studio Code (VSC) is installed on your system.
3. `pip` is installed on your system.

### Step 1: Install `pre-commit`

First, you need to install the `pre-commit` tool on your system. Open your terminal or command prompt and run the following command:

```bash
pip install pre-commit
```

### Step 2: Initialize Pre-Commit for Your Project
After reviewing the `.pre-commit-config.yaml` file, to look for hooks configured initialize pre-commit for the  project. Open your terminal or command prompt, navigate to the root directory of your project, and run the following command:
```bash
pre-commit install
```
Output
```bash
pre-commit installed at .git/hooks/pre-commit
```

git add *.py

git commit -m "Check for code quality and consistency"


Modify load_data.py

insert two lines  
   import sklearn
   import joblib


Check again for consistency and quality code 

    ```bash
    git commit -m "Check for code quality and consistency test"
    
            isort (python)...........................................................Failed
            - hook id: isort
            - files were modified by this hook

            Fixing C:\Users\francisco.torres\Documents\GitHub\MLOps_proj2\load\load_data.py

            autopep8.............................................(no files to check)Skipped
            flake8...............................................(no files to check)Skipped
            autoflake............................................(no files to check)Skipped
            black....................................................................Failed
            - hook id: black
            - files were modified by this hook

            reformatted load\load_data.py

            All done! \u2728 \U0001f370 \u2728
            1 file reformatted.
    ```
The load_data.py has been reformatted, please set load_data.py to Staged Changes and check again!

 ```bash
        >git commit -m "Check for code quality and consistency test 2"      
        isort (python)...........................................................Passed
        autopep8.............................................(no files to check)Skipped
        flake8...............................................(no files to check)Skipped
        autoflake............................................(no files to check)Skipped
        black....................................................................Passed
        [main ff201f1] Check for code quality and consistency test 2
        1 file changed, 31 insertions(+), 19 deletions(-)
 ```
This time we have our code clean and consistent.

## TEST Usage

1. Change to root directory.
2. Run `python mlops_project.py` in the terminal.

## Test API

1. Change to root directory.
2. Run `uvicorn api.main:app --reload` in the terminal.

## Checking endpoints
1. Access `http://127.0.0.1:8000/`, you will see a message like this `"Online Fraud Classifier is all ready to go!"`
2. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:

    ![FastAPI Docs](./imgs/fast-api-docs.png)

3. Try running the classify endpoint by providing some data:
	
    **Request body : FRAUD CASES** 
    ```bash
    {
    "type": 4, 
    "amount": 10000000,
    "oldbalanceOrg": 12930418.44,
    "newbalanceOrg": 2930418.44
    }
    ```
    
    **Request body : NO FRAUD CASES** 
    ```bash
    {
    "type": 3, 
    "amount": 87541.63,
    "oldbalanceOrg": 1925591.38,
    "newbalanceOrg": 2013133.01
    }
    ```



## Directory structure & Cookiecutter
1. You will find a structure provided by Cookiecutter
More info: [cookiecutter](https://cookiecutter.readthedocs.io/)

