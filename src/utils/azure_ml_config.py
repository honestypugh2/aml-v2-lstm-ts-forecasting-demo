"""
Azure ML Workspace Configuration and Setup
"""
import os

import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AzureMLConfig:
    """Azure ML workspace configuration"""

    def __init__(self):
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = os.getenv("AZURE_ML_WORKSPACE")
        self.tenant_id = os.getenv("AZURE_TENANT_ID")

        # Initialize ML Client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name
        )

        self._set_tracking_uri()

    def _set_tracking_uri(self) -> str:
        """
        Fetch the workspace using the MLClient and set the tracking URI to create the
        connection with Azure ML workspace.

        :return: The tracking URI for the MLFlowService
        :rtype: str
        """
        workspace = self.ml_client.workspaces.get(self.workspace_name)  # type: ignore

        if workspace:
            mlflow_tracking_uri = workspace.mlflow_tracking_uri
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                return mlflow_tracking_uri

            self.logger.error("Tracking Id not found.")
            raise ValueError("Tracking Id not found.")
        else:
            self.logger.error("Workspace not found.")
            raise ValueError("Workspace not found.")

    def get_ml_client(self):
        """Get Azure ML client"""
        return self.ml_client

    def validate_config(self):
        """Validate configuration"""
        required_vars = [
            "AZURE_SUBSCRIPTION_ID",
            "AZURE_RESOURCE_GROUP",
            "AZURE_ML_WORKSPACE"
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        print("✅ Azure ML configuration is valid")
        return True
