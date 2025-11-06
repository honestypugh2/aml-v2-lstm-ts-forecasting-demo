"""
Azure ML Pipeline for LSTM Time Series Forecasting
"""
import logging
import os

from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import BuildContext, CommandComponent, Environment
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLOpsPipeline:
    """MLOps pipeline for LSTM model training and deployment"""

    def __init__(self, workspace_config: dict):
        self.ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=workspace_config["subscription_id"],
            resource_group_name=workspace_config["resource_group"],
            workspace_name=workspace_config["workspace_name"]
        )

    def create_environment(self):
        """Create custom environment for training"""
        env_name = "lstm-forecasting-env"

        try:
            # Try to get existing environment
            env = self.ml_client.environments.get(env_name, version="1")
            logger.info(f"Environment {env_name} already exists")
            return env
        except Exception:
            # Create new environment
            logger.info(f"Creating environment {env_name}")

            env = Environment(
                name=env_name,
                description="Environment for LSTM time series forecasting",
                build=BuildContext(
                    path="../../",
                    dockerfile_path="mlops/environments/Dockerfile"
                ),
                version="1"
            )

            env = self.ml_client.environments.create_or_update(env)
            return env

    def create_data_prep_component(self):
        """Create data preparation component"""
        return CommandComponent(
            name="data_preparation",
            display_name="Data Preparation",
            description="Prepare time series data for training",
            inputs={
                "raw_data": Input(type="uri_file"),
                "sequence_length": Input(type="integer", default=60),
                "train_split": Input(type="number", default=0.8),
                "val_split": Input(type="number", default=0.1)
            },
            outputs={
                "processed_data": Output(type="uri_folder")
            },
            code="../../src/data_processing",
            command="python prepare_data.py "
                   "--raw-data ${{inputs.raw_data}} "
                   "--sequence-length ${{inputs.sequence_length}} "
                   "--train-split ${{inputs.train_split}} "
                   "--val-split ${{inputs.val_split}} "
                   "--output-path ${{outputs.processed_data}}",
            environment=self.create_environment()
        )

    def create_training_component(self):
        """Create model training component"""
        return CommandComponent(
            name="model_training",
            display_name="LSTM Model Training",
            description="Train LSTM model for time series forecasting",
            inputs={
                "processed_data": Input(type="uri_folder"),
                "epochs": Input(type="integer", default=100),
                "batch_size": Input(type="integer", default=32),
                "learning_rate": Input(type="number", default=0.001),
                "hidden_size": Input(type="integer", default=50),
                "num_layers": Input(type="integer", default=2)
            },
            outputs={
                "trained_model": Output(type="uri_folder"),
                "metrics": Output(type="uri_file")
            },
            code="../../src/training",
            command="python train_lstm.py "
                   "--data-path ${{inputs.processed_data}} "
                   "--epochs ${{inputs.epochs}} "
                   "--batch-size ${{inputs.batch_size}} "
                   "--learning-rate ${{inputs.learning_rate}} "
                   "--hidden-size ${{inputs.hidden_size}} "
                   "--num-layers ${{inputs.num_layers}} "
                   "--model-output ${{outputs.trained_model}} "
                   "--metrics-output ${{outputs.metrics}}",
            environment=self.create_environment()
        )

    def create_evaluation_component(self):
        """Create model evaluation component"""
        return CommandComponent(
            name="model_evaluation",
            display_name="Model Evaluation",
            description="Evaluate trained LSTM model",
            inputs={
                "trained_model": Input(type="uri_folder"),
                "test_data": Input(type="uri_folder")
            },
            outputs={
                "evaluation_results": Output(type="uri_file")
            },
            code="../../src/evaluation",
            command="python evaluate_model.py "
                   "--model-path ${{inputs.trained_model}} "
                   "--test-data ${{inputs.test_data}} "
                   "--output ${{outputs.evaluation_results}}",
            environment=self.create_environment()
        )

    @pipeline(description="LSTM Time Series Forecasting Pipeline")
    def lstm_training_pipeline(
        self,
        raw_data: Input,
        sequence_length: int = 60,
        train_split: float = 0.8,
        val_split: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        hidden_size: int = 50,
        num_layers: int = 2
    ):
        """Define the training pipeline"""

        # Data preparation step
        data_prep = self.create_data_prep_component()(
            raw_data=raw_data,
            sequence_length=sequence_length,
            train_split=train_split,
            val_split=val_split
        )

        # Model training step
        training = self.create_training_component()(
            processed_data=data_prep.outputs.processed_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        # Model evaluation step
        evaluation = self.create_evaluation_component()(
            trained_model=training.outputs.trained_model,
            test_data=data_prep.outputs.processed_data
        )

        return {
            "trained_model": training.outputs.trained_model,
            "metrics": training.outputs.metrics,
            "evaluation_results": evaluation.outputs.evaluation_results
        }

    def submit_pipeline(
        self, data_path: str, experiment_name: str = "lstm-forecasting"
    ):
        """Submit pipeline for execution"""

        # Create pipeline job
        pipeline_job = self.lstm_training_pipeline(
            raw_data=Input(type="uri_file", path=data_path)
        )

        # Configure job
        pipeline_job.compute = "cpu-cluster"
        pipeline_job.experiment_name = experiment_name

        # Submit job
        job = self.ml_client.jobs.create_or_update(pipeline_job)
        logger.info(f"Pipeline submitted. Job name: {job.name}")

        return job

def main():
    """Main function to create and submit pipeline"""
    # Workspace configuration
    workspace_config = {
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
        "workspace_name": os.getenv("AZURE_ML_WORKSPACE")
    }

    # Initialize pipeline
    mlops_pipeline = MLOpsPipeline(workspace_config)

    # Submit pipeline (replace with actual data path)
    data_path = "azureml://datastores/workspaceblobstore/paths/sample_data.csv"
    job = mlops_pipeline.submit_pipeline(data_path)

    print(f"Pipeline job submitted: {job.name}")
    print(f"Monitor at: {job.studio_url}")

if __name__ == "__main__":
    main()
