"""
Azure ML Utilities for LSTM Time Series Forecasting Project.

This module provides utility functions for Azure ML workspace management,
compute cluster setup, environment configuration, and model deployment.
"""

import logging
import os
from typing import Any, Dict, Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute,
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureMLUtils:
    """Azure ML utilities for workspace and resource management."""

    def __init__(self):
        """Initialize Azure ML client with workspace configuration."""
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

        if not all([self.subscription_id, self.resource_group, self.workspace_name]):
            raise ValueError(
                "Missing Azure ML configuration. Please set AZURE_SUBSCRIPTION_ID, "
                "AZURE_RESOURCE_GROUP, and AZURE_WORKSPACE_NAME in your .env file."
            )

        # Initialize ML client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name
        )

        logger.info(f"Initialized Azure ML client for workspace: {self.workspace_name}")

    def get_or_create_compute(
        self,
        compute_name: str = "cpu-cluster",
        vm_size: str = "Standard_DS3_v2",
        min_instances: int = 0,
        max_instances: int = 4
    ) -> AmlCompute:
        """
        Get existing compute cluster or create a new one.

        Args:
            compute_name: Name of the compute cluster
            vm_size: Azure VM size for compute nodes
            min_instances: Minimum number of nodes
            max_instances: Maximum number of nodes

        Returns:
            AmlCompute: The compute cluster object
        """
        try:
            # Try to get existing compute
            compute = self.ml_client.compute.get(compute_name)
            logger.info(f"Using existing compute cluster: {compute_name}")
            return compute
        except Exception:
            # Create new compute cluster
            logger.info(f"Creating new compute cluster: {compute_name}")

            compute = AmlCompute(
                name=compute_name,
                type="amlcompute",
                size=vm_size,
                min_instances=min_instances,
                max_instances=max_instances,
                idle_time_before_scale_down=300,  # 5 minutes
            )

            compute = self.ml_client.compute.begin_create_or_update(compute).result()
            logger.info(f"Created compute cluster: {compute_name}")
            return compute

    def create_environment(
        self,
        environment_name: str = "lstm-training-env",
        conda_file_path: str = "mlops/environments/ml_environment.yml",
        description: str = "Environment for LSTM time series training"
    ) -> Environment:
        """
        Create or update Azure ML environment.

        Args:
            environment_name: Name of the environment
            conda_file_path: Path to conda environment file
            description: Environment description

        Returns:
            Environment: The Azure ML environment object
        """
        environment = Environment(
            name=environment_name,
            description=description,
            conda_file=conda_file_path,
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )

        environment = self.ml_client.environments.create_or_update(environment)
        logger.info(f"Created/updated environment: {environment_name}")
        return environment

    def register_model(
        self,
        model_name: str,
        model_path: str,
        description: str = "LSTM Time Series Forecasting Model",
        tags: Optional[Dict[str, str]] = None
    ) -> Model:
        """
        Register a model in Azure ML.

        Args:
            model_name: Name for the registered model
            model_path: Local path to the model file
            description: Model description
            tags: Optional tags for the model

        Returns:
            Model: The registered model object
        """
        if tags is None:
            tags = {
                "model_type": "lstm",
                "framework": "pytorch",
                "task": "time_series_forecasting"
            }

        model = Model(
            path=model_path,
            name=model_name,
            description=description,
            tags=tags
        )

        model = self.ml_client.models.create_or_update(model)
        logger.info(f"Registered model: {model_name} (version: {model.version})")
        return model

    def create_online_endpoint(
        self,
        endpoint_name: str,
        description: str = "LSTM Time Series Forecasting Endpoint"
    ) -> ManagedOnlineEndpoint:
        """
        Create a managed online endpoint.

        Args:
            endpoint_name: Name of the endpoint
            description: Endpoint description

        Returns:
            ManagedOnlineEndpoint: The created endpoint
        """
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=description,
            auth_mode="key"
        )

        endpoint = self.ml_client.online_endpoints.begin_create_or_update(
            endpoint
        ).result()
        logger.info(f"Created online endpoint: {endpoint_name}")
        return endpoint

    def deploy_model(
        self,
        endpoint_name: str,
        deployment_name: str,
        model_name: str,
        model_version: str,
        environment_name: str,
        code_path: str = "src/inference",
        scoring_script: str = "score.py",
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1
    ) -> ManagedOnlineDeployment:
        """
        Deploy a model to an online endpoint.

        Args:
            endpoint_name: Name of the target endpoint
            deployment_name: Name of the deployment
            model_name: Name of the registered model
            model_version: Version of the model to deploy
            environment_name: Name of the Azure ML environment
            code_path: Path to inference code
            scoring_script: Name of the scoring script
            instance_type: VM size for deployment
            instance_count: Number of instances

        Returns:
            ManagedOnlineDeployment: The created deployment
        """
        model = f"{model_name}:{model_version}"

        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model,
            environment=environment_name,
            code_configuration=CodeConfiguration(
                code=code_path,
                scoring_script=scoring_script
            ),
            instance_type=instance_type,
            instance_count=instance_count
        )

        deployment = self.ml_client.online_deployments.begin_create_or_update(
            deployment
        ).result()
        logger.info(f"Created deployment: {deployment_name}")

        # Set traffic to 100% for this deployment
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        return deployment

    def get_workspace_info(self) -> Dict[str, Any]:
        """
        Get workspace information and configuration.

        Returns:
            Dict: Workspace details
        """
        workspace = self.ml_client.workspaces.get(self.workspace_name)

        return {
            "name": workspace.name,
            "resource_group": workspace.resource_group,
            "location": workspace.location,
            "subscription_id": self.subscription_id,
            "workspace_id": workspace.id
        }


def get_azure_ml_client() -> MLClient:
    """
    Get configured Azure ML client.

    Returns:
        MLClient: Configured Azure ML client
    """
    utils = AzureMLUtils()
    return utils.ml_client


def setup_workspace() -> Dict[str, Any]:
    """
    Setup Azure ML workspace and return configuration.

    Returns:
        Dict: Workspace configuration and status
    """
    utils = AzureMLUtils()

    # Get workspace info
    workspace_info = utils.get_workspace_info()

    # Setup compute cluster
    compute = utils.get_or_create_compute()

    # Create environment
    environment = utils.create_environment()

    return {
        "workspace": workspace_info,
        "compute": compute.name,
        "environment": environment.name,
        "status": "ready"
    }
