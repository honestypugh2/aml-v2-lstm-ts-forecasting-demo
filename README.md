# Azure ML v2 LSTM Time Series Forecasting Project

## ⚠️ CAUTION: Development Environment Only

**This project is designed for development and learning purposes only with sample/synthetic data.**

For production workloads with actual data, please follow the comprehensive infrastructure, network, and security guidance provided in the official Azure Machine Learning documentation:

- **Enterprise Security**: [Azure Machine Learning Enterprise Security](https://learn.microsoft.com/en-us/azure/machine-learning/concept-enterprise-security?view=azureml-api-2)
- **Secure Workspace Setup**: [Create a Secure Azure ML Workspace](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-create-secure-workspace?view=azureml-api-2)

Production deployments require proper consideration of:
- Network isolation and private endpoints
- Identity and access management
- Data governance and compliance
- Security monitoring and auditing
- Enterprise-grade infrastructure design

## Overview

This project implements an end-to-end LSTM Time Series Forecasting solution using:
- **Azure Machine Learning v2 SDK**
- **PyTorch LSTM models (CPU-only)**
- **MLflow for experiment tracking**
- **uv for fast, cross-platform dependency management**
- **MLOps pipelines and automation**

## 🚀 Key Features

- Complete project structure with modular architecture
- CPU-only PyTorch installation (no CUDA dependencies)
- Cross-platform setup (Linux, macOS, Windows) with uv
- Azure ML workspace integration with proper permissions
- MLflow experiment tracking with v2 compatibility
- Docker environment setup
- Comprehensive testing framework
- Built-in troubleshooting for common Azure ML issues

## 📁 Project Structure

```
aml-v2-lstm-ts-forecasting-demo/
├── README.md                      # Project documentation
├── pyproject.toml                 # Project configuration (PEP 621, CPU-only PyTorch)
├── uv.lock                        # Cross-platform lockfile
├── environment.yml                # Conda environment
├── requirements.in                # Input requirements
├── .env.example                   # Environment variables template
├── setup.sh                       # Environment setup script (Linux/macOS)
├── setup.ps1                      # Environment setup script (Windows)
├── fast_setup.sh                  # Quick setup script (Linux/macOS)
├── fast_setup.ps1                 # Quick setup script (Windows)
├── LICENSE                        # Project license
│
├── infra/                         # Infrastructure as Code (Terraform)
│   ├── terraform/                 # Secure workspace with Managed VNet isolation
│   │   ├── main.tf                # AML workspace, storage, KV, ACR (all firewalled)
│   │   ├── bastion_jumpbox.tf     # Azure Bastion host
│   │   ├── jumpbox_vm.tf          # Windows jumpbox VM (Entra ID joined)
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── providers.tf
│   │   ├── terraform.tfvars.example
│   │   └── README.md
│   ├── terraform-quickstart/      # Minimal public-access workspace
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── providers.tf
│   │   ├── terraform.tfvars.example
│   │   └── README.md
│   └── azure-ml-vnet/             # BYO VNet networking foundation
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       ├── versions.tf
│       ├── terraform.tfvars.example
│       └── README.md
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── azure_ml_training/         # Azure ML training scripts
│   │   ├── train_lstm.py          # Main training script (MLflow v2 compatible)
│   │   ├── submit_training_job.py # Job submission script
│   │   ├── environment.yml        # Conda environment for Azure ML
│   │   ├── requirements.txt       # Python dependencies
│   │   └── README.md              # Training documentation
│   ├── training/                  # Production training framework
│   │   ├── __init__.py
│   │   └── train_lstm.py          # Advanced modular training script
│   ├── data_processing/           # Data preprocessing modules
│   │   ├── __init__.py
│   │   └── preprocessor.py        # Time series preprocessing
│   ├── models/                    # Model definitions
│   │   ├── __init__.py
│   │   ├── lstm_model.py          # PyTorch LSTM implementation
│   │   └── lstm_model_seq2seq.py  # Sequence-to-sequence variant
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── azure_ml_config.py     # Azure ML configuration utilities
│   │   └── azure_ml_utils.py      # Azure ML helper functions
│   └── inference/                 # Inference scripts
│       ├── __init__.py
│       └── score.py               # Model scoring script
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_setup_workspace.ipynb   # Azure ML workspace setup & troubleshooting
│   ├── 02_data_exploration.ipynb  # EDA and preprocessing
│   ├── 03_model_training.ipynb    # Interactive training
│   └── 04_azure_ml_training_tutorial.ipynb # Azure ML training tutorial
│
├── mlops/                         # MLOps configuration
│   ├── pipelines/                 # Azure ML pipelines
│   │   └── training_pipeline.py   # Automated training pipeline
│   ├── environments/              # ML environments
│   │   ├── Dockerfile             # Custom environment
│   │   ├── ml_environment.yml     # Environment specification
│   │   └── requirements.txt       # Environment dependencies
│   └── compute/                   # Compute configurations
│       └── setup_compute.py       # Compute cluster setup
│
├── configs/                       # Configuration files
│   ├── model_config.yaml          # Model hyperparameters
│   └── pipeline_config.yaml       # Pipeline configuration
│
├── docker/                        # Docker configuration
│   ├── Dockerfile                 # ML environment
│   └── requirements.txt           # Docker dependencies
│
├── scripts/                       # Setup and utility scripts
│   ├── fast_setup.sh              # Quick environment setup (Linux/macOS)
│   └── fast_setup.ps1             # Quick environment setup (Windows)
│
├── data/                          # Data directory
│   └── processed/                 # Preprocessed training data
│       ├── train_data.csv, val_data.csv, test_data.csv
│       ├── X_train.npy, X_val.npy, X_test.npy
│       └── y_train.npy, y_val.npy, y_test.npy
│
├── outputs/                       # Model outputs and logs
│   ├── models/
│   │   └── lstm_model.pth         # Trained model checkpoint
│   └── logs/
│
└── tests/                         # Unit tests
    ├── test_model.py              # Model unit tests
    └── test_preprocessor.py       # Preprocessing tests
```

## 🏗️ Infrastructure as Code

The `infra/` directory contains three Terraform deployment options, from quickstart to fully secured:

| Option | Directory | Description |
|--------|-----------|-------------|
| **Quickstart** | `infra/terraform-quickstart/` | Minimal public-access workspace — no VNet, no private endpoints. Ideal for the [Azure ML in a Day](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-azure-ml-in-a-day?view=azureml-api-2) tutorial. |
| **Secure (Managed VNet)** | `infra/terraform/` | Production-grade workspace with Managed VNet isolation, firewalled Storage/KV/ACR, Azure Bastion, and an Entra-joined Windows jumpbox with NAT Gateway. |
| **BYO VNet Foundation** | `infra/azure-ml-vnet/` | Standalone networking layer (VNet, delegated subnet, NSG service-tag rules, route table, managed identity) for Bring Your Own VNet topologies. |

### Quick Start (Quickstart option)

```bash
cd infra/terraform-quickstart
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

az login
terraform init
terraform plan
terraform apply
```

See each subdirectory's README for full architecture diagrams, resource lists, and deployment instructions.

## 🔧 Prerequisites and Setup

### Azure Prerequisites

#### Azure Resources Required
- **Azure Subscription** with active billing
- **Resource Group** containing the ML workspace
- **Azure Machine Learning Workspace** (provisioned via Terraform in `infra/` or created manually via Azure Portal)
- **Storage Account** (automatically created with ML workspace)
- **Key Vault** (automatically created with ML workspace)
- **Application Insights** (automatically created with ML workspace)

#### Azure Roles and Permissions Required

Your Azure account **MUST** have the following roles assigned:

##### At Subscription Level:
- **Contributor** or **Owner** (for resource management)
- **Azure AI User** (built-in RBAC role designed for individuals who need to use AI resources)
- **User Access Administrator** (for role assignments)

##### At Resource Group Level:
- **Contributor** (minimum required)
- **Azure AI User** (inherited from Subscription)
- **AzureML Data Scientist** (recommended for AzureML workspace)
- **AzureML Compute Operator** (recommended for AzureML workspace)
- **Storage Blob Data Contributor** (recommended for Storage Account)

##### Critical: Storage Account Permissions
The Azure ML workspace's **managed identity** needs:
- **Storage Blob Data Contributor** role on the default storage account
- **Storage File Data Privileged Contributor** role on the default storage account

### Local Development Prerequisites

#### Required Software
```bash
# Python 3.11+ required
python --version  # Should show 3.11.x or higher

# Azure CLI (latest version)
az --version

# uv for dependency management (https://docs.astral.sh/uv/)
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv --version
```

#### Environment Variables
Create a `.env` file in the project root by copying `.env.example` and renaming it to `.env`:
```bash
# Azure ML Configuration
AZURE_SUBSCRIPTION_ID="your-subscription-id"
AZURE_RESOURCE_GROUP="your-resource-group"
AZURE_ML_WORKSPACE_NAME="your-workspace-name"
AZURE_TENANT_ID="your-tenant-id"

# Optional: Specific regions
AZURE_LOCATION="eastus2"  # or your preferred region
```

## 🛠️ Installation and Setup

### Step 1: Clone and Install Dependencies

**Linux / macOS:**
```bash
# Clone the repository
git clone <repository-url>
cd aml-v2-lstm-ts-forecasting-demo

# Install dependencies (creates .venv automatically, no CUDA downloads)
uv sync

# Add environment to Jupyter kernelspec to use in Notebook
uv run python -m ipykernel install --user --name .venv
```

**Windows (PowerShell):**
```powershell
# Clone the repository
git clone <repository-url>
cd aml-v2-lstm-ts-forecasting-demo

# Install dependencies (creates .venv automatically, no CUDA downloads)
uv sync

# Add environment to Jupyter kernelspec to use in Notebook
uv run python -m ipykernel install --user --name .venv
```

### Step 2: Configure Azure Authentication

```bash
# Login to Azure using the jumpbox managed identity (no browser required)
az login --identity

# Set your subscription
az account set --subscription "your-subscription-id"

# Verify Azure ML workspace access
az ml workspace show \
    --name "your-workspace-name" \
    --resource-group "your-resource-group"
```

> **Note:** The jumpbox VM has a user-assigned managed identity with
> `Contributor` (resource group) and `AzureML Data Scientist` (workspace)
> roles. Using `az login --identity` avoids corporate Conditional Access
> issues that block browser-based auth flows (`az login`,
> `az login --use-device-code`) on non-Intune-compliant devices.

### Step 3: Set Storage Permissions (Critical - If not already done during Resource Creation)

```bash
# Get workspace managed identity principal ID
WORKSPACE_PRINCIPAL_ID=$(az ml workspace show \
    --name "your-workspace-name" \
    --resource-group "your-resource-group" \
    --query "identity.principal_id" \
    --output tsv)

# Get storage account resource ID
STORAGE_ACCOUNT_ID=$(az storage account list \
    --resource-group "your-resource-group" \
    --query "[?contains(name, 'your-storage-account-name')].id" \
    --output tsv)

# Assign Storage Blob Data Contributor role (CRITICAL FOR JOB SUBMISSION)
az role assignment create \
    --assignee $WORKSPACE_PRINCIPAL_ID \
    --role "Storage Blob Data Contributor" \
    --scope $STORAGE_ACCOUNT_ID

# Verify the role assignment
az role assignment list \
    --assignee $WORKSPACE_PRINCIPAL_ID \
    --scope $STORAGE_ACCOUNT_ID \
    --output table
```

### Step 4: Run Setup Notebook

Open the Jupyter Notebook: notebooks/01_setup_workspace.ipynb

In Jupyter Notebook, select Kernel to be the `.venv` virtual environment that should have been created following Step 1.

The notebook will:
- ✅ Test Azure connectivity
- ✅ Verify workspace access
- ✅ Create/validate compute clusters
- ✅ Setup ML environments
- ✅ Test job submission
- ✅ Provide troubleshooting guidance

## 📖 Usage Guide

### Training Script Options

#### 1. Simple Azure ML Training (`src/azure_ml_training/train_lstm.py`)
- **Purpose**: Quick Azure ML job submission with robust error handling
- **Use Case**: Simple experiments, testing Azure ML functionality, production Azure ML jobs
- **Features**: Self-contained script, MLflow v2 compatibility, synthetic data generation
- **Pros**: Fast setup, minimal dependencies, robust error handling
- **Cons**: Limited to synthetic data, basic preprocessing

#### 2. Production Training Framework (`src/training/train_lstm.py`)
- **Purpose**: Modular, scalable training architecture
- **Use Case**: Production workloads, complex preprocessing, real datasets (Remember to prepare your Subscription and resources for Production, see CAUTION above)
- **Features**: Class-based trainer, external preprocessor, configurable models
- **Pros**: Highly modular, scalable, production-ready, real data support
- **Cons**: More complex setup, requires external modules

#### 3. Interactive Training (Notebooks)
- Use `notebooks/03_model_training.ipynb` for experimentation
- MLflow automatically tracks experiments
- Real-time monitoring in Azure ML Studio

### Daily Development Workflow

```bash
# For Azure ML training jobs
uv run python src/azure_ml_training/train_lstm.py

# For local training experiments
uv run python src/training/train_lstm.py

# For pipeline execution
uv run python mlops/pipelines/training_pipeline.py

# Run tests
uv run pytest tests/ -v
```

## 🚨 Common Issues and Solutions

### Issue: "AuthorizationFailure" during Job Submission
**Cause**: Azure ML workspace managed identity lacks storage permissions  
**Solution**: Run the storage permission commands in Step 3 above

### Issue: Compute Cluster Creation Fails
**Cause**: Insufficient permissions or quota limits  
**Solution**: Ensure Contributor role on resource group and check VM quotas

### Issue: Environment Build Failures
**Cause**: Package conflicts or missing dependencies  
**Solution**: Use curated environments or simplify custom environment dependencies

### Issue: Network Connectivity Problems
**Cause**: Firewall rules or DNS issues  
**Solution**: Ensure Azure services are allowed and test connectivity

## 🚀 Next Steps

### Development Enhancements
1. **Real Data Integration**: Replace synthetic data with actual time series datasets
2. **Model Improvements**: Experiment with different architectures and hyperparameters
3. **Feature Engineering**: Add advanced preprocessing and feature extraction

### Production Scaling
1. **GPU Support**: Add GPU compute targets for larger models
2. **Model Deployment**: Set up real-time inference endpoints
3. **Pipeline Automation**: Deploy automated retraining pipelines
4. **Monitoring**: Implement model monitoring and drift detection

### MLOps Integration
1. **CI/CD Integration**: Connect with Azure DevOps or GitHub Actions
2. **Model Registry**: Implement model versioning and approval workflows
3. **A/B Testing**: Set up model comparison and gradual rollout

## 💡 Best Practices

### Azure ML Integration
- **Storage Permissions**: Always configure managed identity permissions first
- **Compute Management**: Use system-assigned managed identity for clusters
- **Environment Strategy**: Start with curated environments, customize as needed
- **Error Handling**: Implement comprehensive error detection and recovery

### uv + PyTorch Best Practices
- **CPU-First Development**: Use CPU versions for development, GPU for production
- **Clean Dependencies**: Remove unused CUDA packages to avoid bloat
- **Cross-Platform Lock**: `uv.lock` resolves for all platforms by default

### MLflow Integration
- **Centralized Tracking**: Use Azure ML's built-in MLflow tracking
- **Experiment Organization**: Structure experiments with meaningful names and tags
- **Model Registry**: Use MLflow model registry for version control

## 📖 Documentation

- [Azure ML v2 SDK Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [PyTorch Time Series Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [uv Dependency Management](https://docs.astral.sh/uv/)

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check Azure ML Studio**: Look for detailed error messages in the Azure ML portal
2. **Review Activity Logs**: Check Azure Activity Log for resource-level errors
3. **Enable Diagnostics**: Turn on diagnostic logging for Azure ML workspace
4. **Contact Support**: For persistent issues, contact Azure support with:
   - Subscription ID
   - Resource group name
   - Workspace name
   - Correlation ID from error messages
   - Timestamp of the issue

---
