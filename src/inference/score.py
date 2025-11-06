"""
Model inference script for LSTM time series forecasting
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMPredictor:
    """LSTM model predictor for inference"""

    def __init__(self, model_path: str, scaler_path: str | None = None):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model()
        if self.scaler_path and self.scaler_path.exists():
            self.load_scaler()

    def load_model(self):
        """Load trained model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Recreate model from config
            from src.models.lstm_model import LSTMTimeSeriesModel
            config = checkpoint['config']

            self.model = LSTMTimeSeriesModel(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=config['output_size'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional']
            )

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully from {self.model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_scaler(self):
        """Load data scaler"""
        try:
            self.scaler = joblib.load(self.scaler_path) # type: ignore
            logger.info(f"Scaler loaded successfully from {self.scaler_path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise

    def preprocess_input(self, data: List[float],
                         sequence_length: int = 60) -> torch.Tensor:
        """Preprocess input data for prediction"""
        # Convert to numpy array
        data_array = np.array(data).reshape(-1, 1)

        # Scale data if scaler is available
        if self.scaler: # type: ignore
            data_array = self.scaler.transform(data_array) # type: ignore

        # Create sequence
        if len(data_array) < sequence_length: # type: ignore
            raise ValueError(
                f"Input data length ({len(data_array)}) is less than " # type: ignore
                f"required sequence length ({sequence_length})"
            )

        # Take the last sequence_length points
        sequence = data_array[-sequence_length:].flatten() # type: ignore

        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)

        return sequence_tensor.to(self.device)

    def predict(self, input_data: List[float],
                sequence_length: int = 60) -> Dict[str, Any]:
        """Make prediction"""
        try:
            # Preprocess input
            input_tensor = self.preprocess_input(input_data, sequence_length)

            # Make prediction
            with torch.no_grad():
                prediction = self.model(input_tensor) # type: ignore
                prediction_value = prediction.cpu().numpy().flatten()[0]

            # Inverse transform if scaler is available
            if self.scaler: # type: ignore
                prediction_value = self.scaler.inverse_transform( # type: ignore
                    [[prediction_value]]
                )[0][0]

            return {
                "prediction": float(prediction_value), # type: ignore
                "input_length": len(input_data),
                "sequence_length": sequence_length,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }

    def predict_batch(self, batch_data: List[List[float]],
                      sequence_length: int = 60) -> List[Dict[str, Any]]:
        """Make predictions for batch of inputs"""
        results = []
        for data in batch_data:
            result = self.predict(data, sequence_length)
            results.append(result) # type: ignore
        return results # type: ignore

def init():
    """Initialize model for Azure ML endpoint"""
    global model_predictor

    # Get model path from environment or use default
    import os
    model_path = os.getenv("AZUREML_MODEL_DIR", "./outputs/models")
    model_file = Path(model_path) / "lstm_model.pth"
    scaler_file = Path(model_path) / "scaler.pkl"

    model_predictor = LSTMPredictor(
        model_path=str(model_file),
        scaler_path=str(scaler_file) if scaler_file.exists() else None
    )

    logger.info("Model initialized for inference")

def run(raw_data: str) -> str:
    """Run inference for Azure ML endpoint"""
    try:
        # Parse input data
        input_data = json.loads(raw_data)

        # Extract parameters
        data = input_data.get("data", [])
        sequence_length = input_data.get("sequence_length", 60)

        # Make prediction
        if isinstance(data[0], list):
            # Batch prediction
            results = model_predictor.predict_batch(data, sequence_length)
        else:
            # Single prediction
            results = model_predictor.predict(data, sequence_length)

        return json.dumps(results)

    except Exception as e:
        error_response = {
            "error": str(e),
            "status": "error"
        }
        return json.dumps(error_response)

if __name__ == "__main__":
    # Test locally
    sample_data = [100.0 + i + np.random.normal(0, 5) for i in range(100)]

    predictor = LSTMPredictor(
        model_path="./outputs/models/lstm_model.pth",
        scaler_path="./outputs/models/scaler.pkl"
    )

    result = predictor.predict(sample_data)
    print(f"Prediction result: {result}")
