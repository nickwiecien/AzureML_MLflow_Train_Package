# Step 1. Get Data
# Sample Python script designed to load data into a target datastore
# Here we load the Iris Setosa dataset as an example.
# NOTE: REMOVE THIS STEP ENTIRELY WHEN PULLING EXISTING DATA FROM DATASTORES.

from azureml.core import Run, Workspace, Datastore, Dataset
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.datasets import load_iris
import shutil

# Parse input arguments
parser = argparse.ArgumentParser("Add sample Iris Setosa data to default datastore")

args, _ = parser.parse_known_args()

# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Connect to default blob datastore
ds = ws.get_default_datastore()

################################# MODIFY #################################

# The intent of this block is to seed a datastore with data to be used
# for model training downstream. This step should be removed entirely
# if consuming existing data from an existing datastore.

from sklearn.datasets import load_iris
import pandas as pd
import os
import shutil

data = load_iris()

input_df = pd.DataFrame(data.data, columns = data.feature_names)
output_df = pd.DataFrame(data.target, columns = ['target'])

merged_df = pd.concat([input_df, output_df], axis=1)

os.makedirs('./tmp', exist_ok=True)
merged_df.to_csv('./tmp/iris_data.csv', index=False)

ds.upload(src_dir='./tmp',
                 target_path='iris_data_training',
                 overwrite=True)

shutil.rmtree('./tmp')

##########################################################################
