# Azure Machine Learning Demo - Train, Register, and Containerize a MLflow Model
This repository contains an end-to-end Azure Machine Learning pipeline (designed to be published and triggered by an Azure Pipeline [Azure DevOps]) for sourcing data from blob storage, splitting data into test/train subsets, training a classification model and logging metrics using `scikit-learn` and `mlflow`, performing an A/B (champion vs. challenger) test to assess model performance, registration of a newly crowned champion, and finally packaging of a model into a Docker container for deployment to an endpoint.

The steps required to run the code in this sample repo are described below...

## Environment Setup

#### Azure Machine Learning Workspace - Privately Networked
The code contained within this repo is designed to run on compute within an Azure Machine Learning workspace. We recommend provisioning a privately networked AML workspace using the sample template below:
[Create a Secure Azure Machine Learning Workspace using a Template](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-create-secure-workspace-template?tabs=bicep%2Ccli)

#### Self-Hosted Build Agent
Because we are executing Azure Pipelines which need to connect with privately networked resources, it is recommended that you provision a VM to be used as a self-hosted build agent in Azure DevOps. Due to the pre-configured ML environment, we recommend using a Data Science Virtual Machine for this agent.
[What is a Data Science Virtual Machine?](https://learn.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/overview)

Follow the steps in this guide to set up your self-hosted build agent.
[Deploy a Self-Hosted Build Agent in Azure DevOps](https://learn.microsoft.com/en-us/azure/devops/pipelines/agents/v2-linux?view=azure-devops)

Alternatively, you may consider using a VM scale set with a similar private neworking configuration. Details for creating and configuring an Azure VM scale set as a build agent are included int he documentation below.
[Azure virtual machine scale set agents](https://learn.microsoft.com/en-us/azure/devops/pipelines/agents/scale-set-agents?view=azure-devops)

After provisioning this VM, update the networking settings on the provisioned storage account to allow communications from the VNet your VM is deployed into, or whitelist the public IP of this agent.

#### Azure DevOps Project
Within the `mlops_pipelines` subdir is a yaml definition of an Azure Pipeline to publish and trigger the AML pipeline within our targeted workspace. First, create a new DevOps project within your organization, the complete the following steps.

- Create a Service Connection to your Azure ML Workspace
    - Click `Project settings` then drill into `Service connections` and click `New service connection`
    - From the menu that appears select 'Azure resource manager' and click `Next`
    - Choose 'Service principal (automatic)' and click `Next`
    - Under the creation menu, toggle `Machine learning workspace` and enter details about your targeted workspace
    - Name your service connection 'workspace_svc_connection' or something similar and write this name down
    - Click `Next` to complete creation of your service connection

- Create a Variable Group named `azureml_variables` and add the following:

| Variable Name                                | Value                                    |
|-------------------------------------|------------------------------------------|
|SUBSCRIPTION_ID                 | Id of the Azure Subscription which contains your target AML workspacee |
| RESOURCE_GROUP            | Name of the Azure Resource Group which contains your target AML workspace |
| WORKSPACE_NAME | Name of the targeted Azure Machine Learning workspace |
| EXPERIMENT_NAME     | Name to be given to DevOps-pipeline submitted runs |
| WORKSPACE_SVC_CONNECTION     | Name of the created DevOps Service Connection to your target Azure Machine Learning workspace (should be 'workspace_svc_connection') |

- Create a new pipeline
    - From Azure DevOps create a new pipeline
    - Import code from GitHub (either this repository or a forked version)
    - Under pipeline configuration options choose `Existing Azure Pipelines YAML file`
    - When prompted select the yaml file located at `/mlops_pipelines/trigger_train_package.yaml`
    - After reviewing the pipeline definition click `Run`
    
## Code Execution
After configuring your environment and DevOps pipeline as outlined above, you should be able to run your pipeline and watch the code execute in your Azure Machine Learning workspace. 
