"""
Compute cluster setup for Azure ML
"""
import logging
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, ComputeInstance, IdentityConfiguration
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComputeManager:
    """Manage Azure ML compute resources"""

    def __init__(self):
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = os.getenv("AZURE_ML_WORKSPACE")

        # Initialize ML Client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name
        )

    def create_compute_cluster(
        self,
        cluster_name: str = "cpu-cluster",
        vm_size: str = "Standard_DS3_v2",
        min_instances: int = 0,
        max_instances: int = 4,
        idle_time_before_scale_down: int = 120
    ):
        """Create or update compute cluster"""
        try:
            # Check if compute cluster already exists
            existing_cluster = self.ml_client.compute.get(cluster_name)
            logger.info(f"Compute cluster '{cluster_name}' already exists")
            return existing_cluster
        except Exception:
            # Create new compute cluster
            logger.info(f"Creating compute cluster '{cluster_name}'...")

            # Configure managed identity
            identity_config = IdentityConfiguration(
                type="SystemAssigned"
            )


            cluster = AmlCompute(
                name=cluster_name,
                type="amlcompute",
                size=vm_size,
                min_instances=min_instances,
                max_instances=max_instances,
                idle_time_before_scale_down=idle_time_before_scale_down,
                identity=identity_config,
                tier="Dedicated"
            )

            cluster = self.ml_client.compute.begin_create_or_update(cluster).result()
            logger.info(f"Compute cluster '{cluster_name}' created successfully")
            return cluster

    def create_compute_instance(
        self,
        instance_name: str = "compute-instance",
        vm_size: str = "Standard_DS3_v2"
    ):
        """Create compute instance for development"""
        try:
            # Check if compute instance already exists
            existing_instance = self.ml_client.compute.get(instance_name)
            logger.info(f"Compute instance '{instance_name}' already exists")
            return existing_instance
        except Exception:
            # Create new compute instance
            logger.info(f"Creating compute instance '{instance_name}'...")

            instance = ComputeInstance(
                name=instance_name,
                size=vm_size
            )

            instance = self.ml_client.compute.begin_create_or_update(instance).result()
            logger.info(f"Compute instance '{instance_name}' created successfully")
            return instance

    def list_compute_resources(self):
        """List all compute resources"""
        logger.info("Listing compute resources...")

        compute_resources = self.ml_client.compute.list()
        for compute in compute_resources:
            logger.info(f"Name: {compute.name}, Type: {compute.type}, State: {compute.provisioning_state}")  # noqa: E501

        return list(compute_resources)

    def delete_compute_resource(self, compute_name: str):
        """Delete compute resource"""
        try:
            self.ml_client.compute.begin_delete(compute_name).result()
            logger.info(f"Compute resource '{compute_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting compute resource '{compute_name}': {str(e)}")

def main():
    """Main function to setup compute resources"""
    compute_manager = ComputeManager()

    # Create CPU compute cluster for training
    compute_manager.create_compute_cluster(
        cluster_name="cpu-cluster",
        vm_size="Standard_DS3_v2",
        max_instances=4
    )

    # Create GPU compute cluster for training (optional)
    compute_manager.create_compute_cluster(
        cluster_name="gpu-cluster",
        vm_size="Standard_NC6s_v3",
        max_instances=2
    )

    # Create compute instance for development
    compute_manager.create_compute_instance(
        instance_name="dev-compute-instance",
        vm_size="Standard_DS3_v2"
    )

    # List all compute resources
    compute_manager.list_compute_resources()

if __name__ == "__main__":
    main()
