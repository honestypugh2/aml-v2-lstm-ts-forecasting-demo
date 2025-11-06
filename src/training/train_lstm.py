"""
Training script for LSTM time series forecasting model
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data_processing.preprocessor import TimeSeriesPreprocessor, load_sample_data
from models.lstm_model import LSTMConfig, LSTMTimeSeriesModel
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMTrainer:
    """LSTM model trainer with MLflow integration"""

    def __init__(self, config: LSTMConfig, model_save_path: str = "./outputs/models"):
        self.config = config
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = LSTMTimeSeriesModel(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        ).to(self.device)

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Initialize preprocessor
        self.preprocessor = TimeSeriesPreprocessor(
            sequence_length=config.sequence_length
        )

    def prepare_data(self, data_path: str = None) -> tuple: # type: ignore
        """Prepare training data"""
        logger.info("Loading and preparing data...")

        # Load data
        if data_path and os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            logger.info("Using sample data")
            data = load_sample_data()

        # Preprocess data
        scaled_data = self.preprocessor.fit_transform(data)

        # Create sequences
        sequences, targets = self.preprocessor.create_sequences(
            scaled_data, forecast_horizon=1
        )

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data( # type: ignore
            sequences, targets,
            train_size=self.config.train_split,
            val_size=self.config.val_split
        )

        # Create data loaders
        train_loader, val_loader, test_loader = self.preprocessor.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=self.config.batch_size
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Full training loop with MLflow tracking"""

        # Start MLflow run
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "sequence_length": self.config.sequence_length,
                "bidirectional": self.config.bidirectional
            })

            best_val_loss = float('inf')
            train_losses = []
            val_losses = []

            for epoch in range(self.config.num_epochs):
                # Training
                train_loss = self.train_epoch(train_loader)

                # Validation
                val_loss = self.validate(val_loader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, step=epoch)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(epoch, val_loss)

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{self.config.num_epochs} - "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Log final model
            mlflow.pytorch.log_model(self.model, "model")

            return {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss
            }

    def save_model(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_save_path / f"lstm_model_epoch_{epoch}_{timestamp}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }, model_path)

        logger.info(f"Model saved to {model_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train LSTM Time Series Model")
    parser.add_argument("--data-path", type=str, help="Path to training data CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=50,
                        help="Hidden size")
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of LSTM layers"
    )

    args = parser.parse_args()

    # Initialize config
    config = LSTMConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.hidden_size = args.hidden_size
    config.num_layers = args.num_layers

    # Initialize MLflow
    mlflow.set_experiment("lstm-time-series-forecasting")

    # Initialize trainer
    trainer = LSTMTrainer(config)

    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(args.data_path)

    # Train model
    results = trainer.train(train_loader, val_loader) # type: ignore

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()
