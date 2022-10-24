from azureml.core import Workspace, Model
import pandas as pd
import mlflow.sklearn
import numpy as np
import json

model = None

def init():
    global model
    model = mlflow.sklearn.load_model('./deployment/model')

def run(data):
    row_data = json.loads(data)
    df = pd.DataFrame(row_data)
    preds = model.predict(df)
    return [float(x) for x in preds]