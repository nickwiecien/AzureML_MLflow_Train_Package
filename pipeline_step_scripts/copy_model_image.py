# Step 5. Copy Model Image - ALTERNATE
# Sample Python script designed to retrieve an existing
# model image from an Azure Container Registry, tag it
# with a timestamp (locally) and push back to the private ACR.
# This routine establishes access and connectivity to the private ACR. 

from azureml.core import Run, Workspace, Datastore, Dataset
import os
import docker
from datetime import datetime
import argparse

# Parse input arguments
parser = argparse.ArgumentParser("Package model as Docker container")
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_version', type=str, required=True)

args = parser.parse_args()
model_name = args.model_name.lower()
model_version = args.model_version
print(args.model_name)
# Get current run
current_run = Run.get_context()

# Get parent run
parent_run = current_run.parent

#G et associated AML workspace
ws = current_run.experiment.workspace

# Get ACR login creds from Key Vault
kv = ws.get_default_keyvault()
acr_username = kv.get_secret('acr-username')
acr_password = kv.get_secret('acr-password')
acr_address = kv.get_secret('acr-address')

# Get location of existing model image
location = f'{acr_address}/{model_name}:{model_version}'
location

# Create a formatted timestamp as a tag
now = datetime.now()
timestamp = datetime.strftime(now, '%Y%m%d%H%M%S')

# Execute docker commands as using docker python library
# https://pypi.org/project/docker/

# Login to private ACR registry
client = docker.from_env()
client.login(username=acr_username, password=acr_password, registry=acr_address)

# Pull existing model image
img = client.images.pull(location)

# For troubleshooting - print images on current compute
print(client.images.list())

# Tag pulled image with current timestamp (convenient for verifying image creation)
img.tag(f'{acr_address}/{model_name}:{timestamp}')

# Push newly tagged image back to ACR
client.images.push(f'{acr_address}/{model_name}:{timestamp}')