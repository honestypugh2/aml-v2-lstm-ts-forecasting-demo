
#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any

# Import required libraries
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Azure ML imports with error handling
try:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    azure_ml_available = True
    print("✅ Azure ML SDK imported successfully")
except ImportError as e:
    print(f"⚠️ Azure ML SDK import warning: {e}")
    azure_ml_available = False
    # Provide type stubs for when not available
    MLClient = None  # type: ignore
    DefaultAzureCredential = None  # type: ignore

# MLflow imports with error handling
try:
    import mlflow
    import mlflow.pytorch
    mlflow_available = True
    print("✅ MLflow imported successfully")
except ImportError as e:
    print(f"⚠️ MLflow import warning: {e}")
    mlflow_available = False
    # Provide type stubs for when not available
    mlflow = None  # type: ignore


# Simple LSTM model for demonstration
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 50,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(LSTMModel, self).__init__() # type: ignore
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def create_sequences(
    data: np.ndarray[Any, np.dtype[Any]],
    seq_length: int
) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """Create sequences for LSTM training"""
    sequences: list[np.ndarray[Any, np.dtype[Any]]] = []
    targets: list[float] = []

    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def safe_mlflow_log(func_name: str, *args: Any, **kwargs: Any) -> None:
    """Safely log to MLflow with error handling"""
    if not mlflow_available:
        return

    try:
        if func_name == 'log_params':
            mlflow.log_params(*args, **kwargs)  # type: ignore
        elif func_name == 'log_metrics':
            mlflow.log_metrics(*args, **kwargs)  # type: ignore
        elif func_name == 'log_artifact':
            mlflow.log_artifact(*args, **kwargs)  # type: ignore
        elif func_name == 'log_model':
            # Use simplified model logging to avoid tracking_uri issues
            mlflow.pytorch.log_model(*args, **kwargs)  # type: ignore
    except Exception as e:
        print(f"⚠️ MLflow {func_name} warning: {e}")


def set_tracking_uri(
    subscription_id: str | None = None,
    resource_group: str | None = None,
    workspace_name: str | None = None
) -> str | None:
    """
    Set the MLflow tracking URI to connect with Azure ML workspace.

    :param subscription_id: Azure subscription ID
    :param resource_group: Azure resource group name
    :param workspace_name: Azure ML workspace name
    :return: The tracking URI for MLflow
    :rtype: str
    """
    if not azure_ml_available or not mlflow_available:
        print("⚠️ Azure ML SDK or MLflow not available, skipping tracking URI setup")
        return None

    try:
        # Get credentials and create ML client
        credential = DefaultAzureCredential()  # type: ignore

        # Use environment variables if parameters not provided
        subscription_id = subscription_id or os.environ.get('AZURE_SUBSCRIPTION_ID')
        resource_group = resource_group or os.environ.get('AZURE_RESOURCE_GROUP')
        workspace_name = workspace_name or os.environ.get('AZURE_ML_WORKSPACE_NAME')

        if not all([subscription_id, resource_group, workspace_name]):
            print("⚠️ Missing Azure ML workspace configuration, "
                  "skipping tracking URI setup")
            return None

        ml_client = MLClient(  # type: ignore
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group
        )

        # Get workspace and set tracking URI
        workspace = ml_client.workspaces.get(workspace_name)  # type: ignore

        if workspace and hasattr(workspace, 'mlflow_tracking_uri'):
            mlflow_tracking_uri = workspace.mlflow_tracking_uri
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)  # type: ignore
                print(f"✅ MLflow tracking URI set: {mlflow_tracking_uri}")
                return mlflow_tracking_uri
            else:
                print("⚠️ MLflow tracking URI not found in workspace")
                return None
        else:
            print("⚠️ Workspace not found or doesn't have MLflow tracking URI")
            return None

    except Exception as e:
        print(f"⚠️ Error setting MLflow tracking URI: {e}")
        return None


def generate_sample_data(num_points: int = 1000) -> pd.DataFrame:
    """Generate sample time series data"""
    np.random.seed(42)
    time = np.arange(num_points)

    # Create a time series with trend and seasonality
    trend = 0.02 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 50)
    noise = np.random.normal(0, 2, num_points)

    data = trend + seasonal + noise + 100

    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=num_points, freq='D'),
        'value': data
    })

    return df

def main():
    """Main training function"""
    global mlflow_available

    parser = argparse.ArgumentParser(description='LSTM Training Script')
    parser.add_argument(
        '--sequence_length', type=int, default=10,
        help='Sequence length for LSTM'
    )
    parser.add_argument(
        '--hidden_size', type=int, default=50, help='LSTM hidden size'
    )
    parser.add_argument(
        '--num_layers', type=int, default=2,
        help='Number of LSTM layers'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.2, help='Dropout rate'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs', type=int, default=50, help='Number of epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help='Batch size'
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--subscription_id', type=str, default=None,
        help='Azure subscription ID '
             '(uses AZURE_SUBSCRIPTION_ID env var if not provided)'
    )
    parser.add_argument(
        '--resource_group', type=str, default=None,
        help='Azure resource group (uses AZURE_RESOURCE_GROUP env var if not provided)'
    )
    parser.add_argument(
        '--workspace_name', type=str, default=None,
        help='Azure ML workspace name '
             '(uses AZURE_ML_WORKSPACE_NAME env var if not provided)'
    )

    args = parser.parse_args()

    print("🚀 Starting LSTM Training")
    print(f"📋 Configuration: {vars(args)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set MLflow tracking URI for Azure ML workspace
    print("🔗 Setting up MLflow tracking URI...")
    set_tracking_uri(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )

    # # Start MLflow run with error handling
    # # MLflow imports with error handling
    # try:
    #     import mlflow
    #     import mlflow.pytorch
    #     mlflow_available = True
    #     print("✅ MLflow imported successfully")
    # except ImportError as e:
    #     print(f"⚠️ MLflow import warning: {e}")
    #     mlflow_available = False

    if mlflow_available:
        try:
            mlflow.start_run()  # type: ignore
            print("✅ MLflow run started")
        except Exception as e:
            print(f"⚠️ MLflow start_run warning: {e}")
            mlflow_available = False

    try:
        # Log hyperparameters
        safe_mlflow_log('log_params', {
            'sequence_length': args.sequence_length,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        })

        print("📊 Generating sample data...")
        # Generate or load data
        data = generate_sample_data(1000)
        print(f"Data shape: {data.shape}")

        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['value']]) # type: ignore

        # Create sequences
        sequences, targets = create_sequences(
            scaled_data.flatten(), args.sequence_length # type: ignore
        )

        # Split data
        train_size = int(0.8 * len(sequences))
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]
        val_sequences = sequences[train_size:]
        val_targets = targets[train_size:]

        # Convert to tensors
        train_sequences = torch.FloatTensor(train_sequences).unsqueeze(-1)
        train_targets = torch.FloatTensor(train_targets)
        val_sequences = torch.FloatTensor(val_sequences).unsqueeze(-1)
        val_targets = torch.FloatTensor(val_targets)

        # Create data loaders
        train_dataset = TensorDataset(train_sequences, train_targets)
        val_dataset = TensorDataset(val_sequences, val_targets)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        print("🏗️ Creating model...")
        # Create model
        model = LSTMModel(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Model initialized with {num_params} parameters")

        print(f"🏃‍♂️ Training for {args.epochs} epochs...")
        # Training loop
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(args.epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_sequences, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs.squeeze(), batch_targets)
                loss.backward()
                optimizer.step() # type: ignore
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_sequences, batch_targets in val_loader:
                    outputs = model(batch_sequences)
                    loss = criterion(outputs.squeeze(), batch_targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Log metrics with error handling
            safe_mlflow_log('log_metrics', {
                'train_loss': train_loss,
                'val_loss': val_loss
            }, step=epoch)

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch+1}/{args.epochs}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}'
                )

        print("💾 Saving model and artifacts...")
        # Save model
        model_path = os.path.join(args.output_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

        # Save scaler
        scaler_path = os.path.join(args.output_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path) # type: ignore

        # Save training history
        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'hyperparameters': vars(args)
            }, f)

        # Log final metrics
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]

        safe_mlflow_log('log_metrics', {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss
        })

        # Log artifacts with error handling
        safe_mlflow_log('log_artifact', model_path)
        safe_mlflow_log('log_artifact', scaler_path)
        safe_mlflow_log('log_artifact', history_path)

        # Log model with simplified approach
        try:
            if mlflow_available:
                mlflow.pytorch.log_model(  # type: ignore
                    model, "pytorch_model", registered_model_name=None
                )
        except Exception as e:
            print(f"⚠️ Model logging warning: {e}")
            print("Model saved locally to outputs directory")

        print("✅ Training completed successfully!")
        print("📊 Final Results:")
        print(f"   Train Loss: {final_train_loss:.6f}")
        print(f"   Validation Loss: {final_val_loss:.6f}")
        print(f"   Model saved to: {model_path}")

        # Write success marker
        with open(os.path.join(args.output_dir, "SUCCESS"), 'w') as f:
            f.write("Training completed successfully\n")
            f.write(f"Final train loss: {final_train_loss:.6f}\n")
            f.write(f"Final val loss: {final_val_loss:.6f}\n")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()

        # Write error marker
        with open(os.path.join(args.output_dir, "ERROR"), 'w') as f:
            f.write(f"Training failed: {str(e)}\n")

        raise

    finally:
        # End MLflow run with error handling
        if mlflow_available:
            try:
                mlflow.end_run()  # type: ignore
                print("✅ MLflow run ended")
            except Exception as e:
                print(f"⚠️ MLflow end_run warning: {e}")

if __name__ == "__main__":
    main()
