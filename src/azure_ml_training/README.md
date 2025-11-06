# Azure ML Training Script for LSTM Time Series Forecasting

This directory contains Azure ML optimized training scripts for LSTM time series forecasting models.

## Files Overview

- `train_lstm_azureml.py` - Main training script optimized for Azure ML remote execution
- `submit_training_job.py` - Job submission script for Azure ML
- `environment.yml` - Conda environment specification
- `requirements.txt` - Pip requirements file

## Features

### 🚀 Azure ML Optimized Training Script

The `train_lstm_azureml.py` script includes:

- **Self-contained LSTM model** - No external dependencies on local modules
- **Comprehensive argument parsing** - Full control over hyperparameters via CLI
- **MLflow integration** - Automatic experiment tracking and model registration
- **Proper output handling** - Saves to `./outputs` directory as expected by Azure ML
- **Error handling and logging** - Production-ready error handling
- **Model checkpointing** - Saves best models during training
- **Early stopping** - Prevents overfitting with configurable patience
- **Learning rate scheduling** - Adaptive learning rate based on validation loss
- **Comprehensive metrics** - MAE, MSE, RMSE, MAPE tracking

### 📊 Training Features

- **Synthetic data generation** - Built-in sample data for quick testing
- **Data preprocessing** - MinMax scaling and sequence creation
- **Train/validation/test splits** - Proper data splitting
- **GPU support** - Automatic CUDA detection and usage
- **Gradient clipping** - Prevents exploding gradients
- **Model artifacts** - Saves model, scaler, and training history

## Usage

### Option 1: Direct Script Execution

```bash
python train_lstm_azureml.py \
    --hidden-size 64 \
    --num-layers 2 \
    --dropout 0.2 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-epochs 50 \
    --sequence-length 60 \
    --early-stopping-patience 10
```

### Option 2: Azure ML Job Submission

```bash
python submit_training_job.py \
    --compute-name cpu-cluster \
    --experiment-name lstm-forecasting \
    --hidden-size 64 \
    --num-epochs 50
```

### Option 3: From Notebook

Use the notebook cells we created in `01_setup_workspace.ipynb` to submit jobs programmatically.

## Environment Variables

Set these environment variables for Azure ML authentication:

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_ML_WORKSPACE="your-workspace-name"
```

## Available Arguments

### Model Architecture
- `--hidden-size` (int, default: 64) - LSTM hidden size
- `--num-layers` (int, default: 2) - Number of LSTM layers
- `--dropout` (float, default: 0.2) - Dropout rate
- `--input-size` (int, default: 1) - Input feature size
- `--output-size` (int, default: 1) - Output size

### Training Parameters
- `--learning-rate` (float, default: 0.001) - Learning rate
- `--weight-decay` (float, default: 1e-5) - Weight decay for regularization
- `--batch-size` (int, default: 32) - Batch size
- `--num-epochs` (int, default: 100) - Number of training epochs
- `--sequence-length` (int, default: 60) - Input sequence length

### Data Parameters
- `--data-path` (str) - Path to CSV file with time series data
- `--num-data-points` (int, default: 1000) - Synthetic data points if no data provided
- `--train-split` (float, default: 0.7) - Training data ratio
- `--val-split` (float, default: 0.15) - Validation data ratio

### Training Control
- `--early-stopping-patience` (int, default: 15) - Early stopping patience
- `--scheduler-patience` (int, default: 7) - Learning rate scheduler patience

## Outputs

The training script creates the following outputs in the `./outputs` directory:

```
outputs/
├── models/
│   ├── best_model.pth          # Best model checkpoint
│   ├── latest_checkpoint.pth   # Latest model checkpoint
│   └── scaler.joblib          # Data scaler for inference
└── metrics/
    └── training_history.json  # Training metrics and history
```

## Azure ML Integration

### Environment

The script uses a conda environment with:
- Python 3.9
- PyTorch >= 1.12.0
- MLflow >= 2.0.0
- Azure ML packages
- Standard ML libraries

### MLflow Tracking

Automatically logs:
- **Hyperparameters** - All model and training parameters
- **Metrics** - Loss, MAE, MSE, RMSE, MAPE per epoch
- **Artifacts** - Model files, scaler, training history
- **Model** - Registered PyTorch model for deployment

### Compute Requirements

- **CPU clusters** - Works well with Standard_D-series VMs
- **GPU clusters** - Supports CUDA for faster training
- **Memory** - Minimum 4GB RAM recommended

## Example Workflows

### 1. Quick Test Run
```bash
python train_lstm_azureml.py --num-epochs 10 --batch-size 16
```

### 2. Production Training
```bash
python submit_training_job.py \
    --compute-name gpu-cluster \
    --hidden-size 128 \
    --num-layers 3 \
    --num-epochs 100 \
    --batch-size 64 \
    --early-stopping-patience 20
```

### 3. Custom Data Training
```bash
python train_lstm_azureml.py \
    --data-path /path/to/your/data.csv \
    --sequence-length 30 \
    --hidden-size 64 \
    --num-epochs 50
```

## Monitoring Training

### Azure ML Studio
- View real-time logs and metrics
- Monitor resource usage
- Download outputs and artifacts

### MLflow UI
```bash
mlflow ui --backend-store-uri azureml://your-workspace-uri
```

### Azure CLI
```bash
az ml job show --name your-job-name
az ml job download --name your-job-name --output-name model_output
```

## Troubleshooting

### Common Issues

1. **Environment not found**
   - Ensure environment.yml is in the same directory
   - Check conda channel availability

2. **Compute cluster not available**
   - Verify compute cluster name and status
   - Check quota and resource availability

3. **Authentication errors**
   - Ensure Azure CLI is logged in: `az login`
   - Verify workspace access permissions

4. **Out of memory errors**
   - Reduce batch size
   - Use smaller model (fewer layers/hidden units)
   - Use CPU if GPU memory insufficient

### Performance Tips

1. **Faster training**
   - Use GPU compute targets
   - Increase batch size (if memory allows)
   - Use fewer epochs with early stopping

2. **Better models**
   - Tune sequence length based on your data patterns
   - Experiment with different architectures
   - Use learning rate scheduling

3. **Cost optimization**
   - Use low-priority compute for long experiments
   - Stop compute when not in use
   - Use smaller VM sizes for development

## Next Steps

1. **Model Deployment**
   - Register best model in Azure ML
   - Create online/batch endpoints
   - Set up model monitoring

2. **Pipeline Automation**
   - Create Azure ML pipelines
   - Set up scheduled retraining
   - Implement CI/CD for model updates

3. **Production Integration**
   - Add data validation
   - Implement A/B testing
   - Set up model drift monitoring