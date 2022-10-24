# Step 4. Evaluate and Register Model
# Sample Python script designed to evaluate a newly-trained
# challenger model against a previously-trained "champion"
# model (i.e., the best-performing model to date). In the case
# that the challenger performs better, it should be registered in the
# workspace, otherwise the run should end gracefully.

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
import shutil
import sklearn
import joblib
import numpy as np
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score

# Parse input arguments
parser = argparse.ArgumentParser("Evaluate classified and register if more performant")
parser.add_argument('--target_column', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_description', type=str, required=True)

args, _ = parser.parse_known_args()
target_column = args.target_column
model_name = args.model_name
model_description = args.model_description

# Get current run
current_run = Run.get_context()

# Get parent run
parent_run = current_run.parent

#Get associated AML workspace
ws = current_run.experiment.workspace

#Get default datastore
ds = ws.get_default_datastore()

# Get training/testing datasets
# Both are associated with registered models.
training_dataset = current_run.input_datasets['Training_Data']
testing_dataset = current_run.input_datasets['Testing_Data']
formatted_datasets = [('Training_Data', training_dataset), ('Testing_Data', testing_dataset)]

# Load test dataset
X_test_dataset = current_run.input_datasets['Testing_Data']
X_test = X_test_dataset.to_pandas_dataframe().astype(np.float64)

# Split into X and y components
y_test = X_test[[target_column]]
X_test = X_test.drop(target_column, axis=1)

################################# MODIFY #################################

# The intent of this block is to load the newly-trained challenger model
# and current champion model (if it exists), and evaluate performance of
# both against a common test dataset using a target metric of interest.
# If the challenger performs better, it should be added to the model
# registry.

# Get previous Mlflow Run ID
mlflow_run_id = parent_run.get_tags()['MLflow_Run_ID']
challenger_run = mlflow.get_run(run_id=mlflow_run_id)

# Check if a current version of the model exists in the registry
client = MlflowClient()
model_exists = False
for mv in client.search_model_versions(f"name='{model_name}'"):
    model_exists = True
    
# Load challenger model
challenger_model = mlflow.sklearn.load_model(challenger_run.info.artifact_uri + '/model')
    
if not model_exists:
    # If no registered model exists, register the current model by default
    mlflow.register_model(
        f"runs:/{mlflow_run_id}/model",
        model_name
    )
else:
    # A/B Test champion and challenger
    champion_model = mlflow.sklearn.load_model(f'models:/{model_name}/latest')
    
    challenger_preds = challenger_model.predict(X_test)
    challenger_accuracy = accuracy_score(y_test, challenger_preds)
    
    champion_preds= champion_model.predict(X_test)
    champion_accuracy = accuracy_score(y_test, champion_preds)
    
    if champion_accuracy > challenger_accuracy:
        # Previous model performs better, cancel out
        # Here we proceed by default for demonstration purposes
        # parent_run.cancel()
        pass
    else:
        mlflow.register_model(
            f"runs:/{mlflow_run_id}/model",
            model_name
        )

##########################################################################