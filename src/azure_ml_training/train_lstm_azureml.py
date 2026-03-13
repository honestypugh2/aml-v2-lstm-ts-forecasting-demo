
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow
import mlflow.pytorch
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json


class LSTMTimeSeriesModel(nn.Module):
    """LSTM model for time series forecasting"""

    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMTimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Apply dropout and linear layer to the last output
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out


def generate_synthetic_data(n_samples=1000, sequence_length=50):
    """Generate synthetic time series data"""
    print(f"🔄 Generating {n_samples} samples with sequence length {sequence_length}")

    # Generate time series with trend and seasonality
    t = np.linspace(0, 4*np.pi, n_samples + sequence_length)

    # Create complex time series
    trend = 0.01 * t
    seasonal = 2 * np.sin(t) + 0.5 * np.sin(3*t)
    noise = 0.1 * np.random.randn(len(t))

    data = trend + seasonal + noise

    # Create sequences
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])

    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y).reshape(-1, 1)

    # Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"✅ Data generated - Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test, config):
    """Train the LSTM model with MLflow tracking"""

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Initialize model
    model = LSTMTimeSeriesModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size'],
        dropout=config['dropout']
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    print(f"🚀 Starting training for {config['epochs']} epochs")

    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)

                # Log to MLflow
                mlflow.log_metric("train_loss", loss.item(), step=epoch)
                mlflow.log_metric("test_loss", test_loss.item(), step=epoch)

                print(f"Epoch [{epoch+1}/{config['epochs']}] - "
                      f"Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")

            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).numpy()
        test_pred = model(X_test_tensor).numpy()

        # Calculate final metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        # Log final metrics
        mlflow.log_metric("final_train_mse", train_mse)
        mlflow.log_metric("final_test_mse", test_mse)
        mlflow.log_metric("final_train_mae", train_mae)
        mlflow.log_metric("final_test_mae", test_mae)

        print(f"\n📊 Final Results:")
        print(f"   Train MSE: {train_mse:.6f}")
        print(f"   Test MSE: {test_mse:.6f}")
        print(f"   Train MAE: {train_mae:.6f}")
        print(f"   Test MAE: {test_mae:.6f}")

    return model, test_mse


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='LSTM Time Series Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=50, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--sequence_length', type=int, default=50, help='Input sequence length')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of data samples')

    args = parser.parse_args()

    # Configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'sequence_length': args.sequence_length,
        'n_samples': args.n_samples,
        'input_size': 1,
        'output_size': 1
    }

    print("🎯 Starting Azure ML LSTM Training")
    print(f"📋 Configuration: {json.dumps(config, indent=2)}")

    # Set up MLflow
    mlflow.start_run()

    try:
        # Log parameters
        mlflow.log_params(config)

        # Generate data
        X_train, X_test, y_train, y_test = generate_synthetic_data(
            n_samples=config['n_samples'],
            sequence_length=config['sequence_length']
        )

        # Train model
        model, test_mse = train_model(X_train, y_train, X_test, y_test, config)

        # Save model
        model_path = "lstm_model"
        mlflow.pytorch.log_model(model, model_path)

        # Create model info file
        model_info = {
            "model_type": "LSTM Time Series",
            "framework": "PyTorch",
            "final_test_mse": float(test_mse),
            "parameters": config
        }

        # Save model info
        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        mlflow.log_artifact("model_info.json")

        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to MLflow")
        print(f"🎯 Final Test MSE: {test_mse:.6f}")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
        raise

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
