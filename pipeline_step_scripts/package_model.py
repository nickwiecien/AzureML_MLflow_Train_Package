# Step 5. Package Model
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

# Parse input arguments
parser = argparse.ArgumentParser("Package model as Docker container")
parser.add_argument('--model_name', type=str, required=True)


args = parser.parse_args()
model_name = args.model_name.lower()
print(args.model_name)
# Get current run
current_run = Run.get_context()

# Get parent run
parent_run = current_run.parent

#G et associated AML workspace
ws = current_run.experiment.workspace

model = Model(ws, model_name)
model_version = model.version
print(model)

model.download('./deployment', exist_ok=True)

from azureml.core import Environment
from azureml.core.model import InferenceConfig

env = Environment.get(ws, 'sample_env')
env.register(ws)
inference_config = InferenceConfig(
    environment=env,
    source_directory="./deployment",
    entry_script="./score.py",
)

package = Model.package(ws, [model], inference_config, image_name=model_name, image_label=model_version)
package.wait_for_creation(show_output=True)
location = package.location