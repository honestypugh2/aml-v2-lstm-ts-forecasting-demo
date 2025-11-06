"""
Azure ML Job Submission Script

This script submits the LSTM training job to Azure ML compute cluster.
It demonstrates how to:
1. Create and register environments
2. Submit training jobs with proper configuration
3. Monitor job progress
4. Handle job outputs and artifacts

Usage:
    python submit_training_job.py \
        --compute-name cpu-cluster \
        --experiment-name lstm-forecasting
"""

import argparse
import os
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def create_ml_client(
    subscription_id: str, resource_group: str, workspace_name: str
) -> MLClient:
    """Create Azure ML client with managed identity authentication"""
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    print(f"✅ Connected to Azure ML workspace: {workspace_name}")
    return ml_client


def create_or_get_environment(
    ml_client: MLClient, environment_name: str = "pytorch-lstm-env"
) -> Environment:
    """Create or retrieve Azure ML environment"""

    # Check if environment exists
    try:
        existing_env = ml_client.environments.get(environment_name, label="latest")
        print(
            f"✅ Using existing environment: "
            f"{environment_name}:{existing_env.version}"
        )
        return existing_env
    except Exception:
        print(f"Environment {environment_name} not found, creating new one...")

    # Get the directory containing this script
    script_dir = Path(__file__).parent

    # Create environment from conda file
    environment = Environment(
        name=environment_name,
        description="PyTorch environment for LSTM time series forecasting",
        conda_file=script_dir / "environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    # Register environment
    registered_env = ml_client.environments.create_or_update(environment)
    print(f"✅ Created environment: {registered_env.name}:{registered_env.version}")

    return registered_env


def submit_training_job(
    ml_client: MLClient,
    environment: Environment,
    compute_name: str,
    experiment_name: str,
    display_name: str = None,
    **training_args
):
    """Submit training job to Azure ML"""

    # Get the directory containing this script
    script_dir = Path(__file__).parent

    # Build command with training arguments
    base_command = "python train_lstm_azureml.py"

    # Add training arguments
    command_args = []
    for key, value in training_args.items():
        if value is not None:
            arg_name = key.replace('_', '-')
            command_args.append(f"--{arg_name} {value}")

    full_command = f"{base_command} {' '.join(command_args)}"

    print(f"Training command: {full_command}")

    # Create default display name
    default_display_name = (
        f"LSTM Training - {training_args.get('num_epochs', 100)} epochs"
    )

    # Create job
    job = command(
        code=str(script_dir),  # Source code directory
        command=full_command,
        environment=f"{environment.name}:{environment.version}",
        compute=compute_name,
        experiment_name=experiment_name,
        display_name=display_name or default_display_name,
        description="LSTM time series forecasting training on Azure ML",
        tags={
            "model_type": "LSTM",
            "framework": "PyTorch",
            "task": "time_series_forecasting",
            "training_type": "remote"
        }
    )

    # Submit job
    submitted_job = ml_client.jobs.create_or_update(job)

    print("✅ Job submitted successfully!")
    print(f"   Job name: {submitted_job.name}")
    print(f"   Status: {submitted_job.status}")
    print(f"   Studio URL: {submitted_job.studio_url}")

    return submitted_job


def main():
    """Main function to submit Azure ML training job"""
    parser = argparse.ArgumentParser(
        description="Submit LSTM training job to Azure ML"
    )

    # Azure ML configuration
    parser.add_argument(
        "--subscription-id",
        type=str,
        default=os.getenv("AZURE_SUBSCRIPTION_ID"),
        help="Azure subscription ID"
    )
    parser.add_argument(
        "--resource-group",
        type=str,
        default=os.getenv("AZURE_RESOURCE_GROUP"),
        help="Azure resource group name"
    )
    parser.add_argument(
        "--workspace-name",
        type=str,
        default=os.getenv("AZURE_ML_WORKSPACE"),
        help="Azure ML workspace name"
    )
    parser.add_argument(
        "--compute-name",
        type=str,
        default="cpu-cluster",
        help="Azure ML compute cluster name"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="lstm-time-series-forecasting",
        help="Azure ML experiment name"
    )
    parser.add_argument(
        "--environment-name",
        type=str,
        default="pytorch-lstm-env",
        help="Azure ML environment name"
    )

    # Training hyperparameters (passed to training script)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=10)

    args = parser.parse_args()

    # Validate required arguments
    if not all([args.subscription_id, args.resource_group, args.workspace_name]):
        print("❌ Missing required Azure configuration. Please provide:")
        print("   - subscription-id (or set AZURE_SUBSCRIPTION_ID env var)")
        print("   - resource-group (or set AZURE_RESOURCE_GROUP env var)")
        print("   - workspace-name (or set AZURE_ML_WORKSPACE env var)")
        return

    try:
        # Create ML client
        ml_client = create_ml_client(
            args.subscription_id,
            args.resource_group,
            args.workspace_name
        )

        # Create or get environment
        environment = create_or_get_environment(ml_client, args.environment_name)

        # Prepare training arguments
        training_args = {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "sequence_length": args.sequence_length,
            "early_stopping_patience": args.early_stopping_patience,
        }

        # Submit job
        job = submit_training_job(
            ml_client=ml_client,
            environment=environment,
            compute_name=args.compute_name,
            experiment_name=args.experiment_name,
            **training_args
        )

        print("\n" + "="*50)
        print("🚀 Training job submitted to Azure ML!")
        print("="*50)
        print(f"Job Name: {job.name}")
        print(f"Experiment: {args.experiment_name}")
        print(f"Compute: {args.compute_name}")
        print(f"Studio URL: {job.studio_url}")
        print("\n💡 You can monitor the job progress in Azure ML Studio or use:")
        print(f"   az ml job show --name {job.name}")

    except Exception as e:
        print(f"❌ Error submitting job: {str(e)}")
        raise


if __name__ == "__main__":
    main()
