# Step 3. Train Model
# Sample Python script designed to train a K-Neighbors classification
# model using the Scikit-Learn library.

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse
import shutil

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score, roc_auc_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import neighbors
import joblib
from numpy.random import seed
import mlflow
import mlflow.sklearn


# Parse input arguments
parser = argparse.ArgumentParser("Train classification model")
parser.add_argument('--target_column', type=str, required=True)

args, _ = parser.parse_known_args()
target_column = args.target_column

# Get current run
current_run = Run.get_context()

# Get parent run
parent_run = current_run.parent

#G et associated AML workspace
ws = current_run.experiment.workspace

# Read input dataset to pandas dataframe
X_train_dataset = current_run.input_datasets['Training_Data']
X_train = X_train_dataset.to_pandas_dataframe().astype(np.float64)
X_test_dataset = current_run.input_datasets['Testing_Data']
X_test = X_test_dataset.to_pandas_dataframe().astype(np.float64)

# Split into X and y 
y_train = X_train[[target_column]]
y_test = X_test[[target_column]]

X_train = X_train.drop(target_column, axis=1)
X_test = X_test.drop(target_column, axis=1)

################################# MODIFY #################################

# The intent of this block is to scale data appropriately and train
# a predictive model. Any normalizaton and training approach can be used.
# For simplicity, here we utilize MLflow for run logging and Scikit-Learn
# pipelines for coupling together a scaler and a model - this simplifies 
# downstream inferencing.
# The model is saved in the run artifacts but not registered. In a downstream
# step we will load this model into code and test it against the current 'champion'
# model. Here, we update the parent run with the MLflow Run ID.
import mlflow.sklearn

mlflow.sklearn.autolog()

with mlflow.start_run(run_name=current_run.name) as run:
    run_id = run.info.run_id

    scaler = preprocessing.MinMaxScaler()
    clf = neighbors.KNeighborsClassifier()
    pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])
    pipeline.fit(X_train, y_train)

parent_run.tag('MLflow_Run_ID', run_id)


##########################################################################