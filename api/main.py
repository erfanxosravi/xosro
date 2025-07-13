from fastapi import FastAPI
import torch
import numpy as np
from pydantic import BaseModel
from modeling.hybrid_model import HybridModel # Assuming the model is in the parent directory

# --- Application Setup ---
app = FastAPI(
    title="Financial Advisor API",
    description="An intelligent financial advisor for the Iranian stock market.",
    version="0.1.0",
)

# --- Model Loading ---
# Load the pre-trained model (for now, with the same architecture as in training)
# In a real application, you would load a saved model state_dict
model = HybridModel(
    input_size=5,
    lstm_hidden_size=50,
    lstm_num_layers=2,
    mlp_hidden_size=25,
    output_size=2
)
# model.load_state_dict(torch.load("path/to/your/saved_model.pth")) # Example of loading a real model
model.eval() # Set the model to evaluation mode

# --- API Data Models ---
class PredictionRequest(BaseModel):
    # Represents the data required for a single prediction
    historical_data: list[list[float]] # e.g., [[open, high, low, close, volume], ...]
    sentiment_score: float

    class Config:
        schema_extra = {
            "example": {
                "historical_data": [
                    [100, 102, 99, 101, 15000],
                    [101, 103, 100, 102, 18000],
                    # ... (10 days of data)
                ],
                "sentiment_score": 0.75
            }
        }

class PredictionResponse(BaseModel):
    growth_probability: float
    fall_probability: float
    suggestion: str

# --- API Endpoints ---
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predicts the market status and provides a suggestion.
    """
    # 1. Preprocess the input data
    # In a real scenario, you'd apply the same scaling as in training
    historical_data_np = np.array([request.historical_data]) # Add batch dimension
    sentiment_score_np = np.array([[request.sentiment_score]])

    # Convert to PyTorch tensors
    time_series_tensor = torch.from_numpy(historical_data_np).float()
    sentiment_tensor = torch.from_numpy(sentiment_score_np).float()

    # 2. Get model prediction
    with torch.no_grad():
        prediction = model(time_series_tensor, sentiment_tensor)

    probabilities = prediction[0].tolist()
    growth_prob = probabilities[1] # Assuming index 1 is for 'growth'
    fall_prob = probabilities[0]   # Assuming index 0 is for 'fall'

    # 3. Generate suggestion based on probabilities
    suggestion = "Hold"
    if growth_prob > 0.7:
        suggestion = "Buy"
    elif fall_prob > 0.7:
        suggestion = "Sell"

    return PredictionResponse(
        growth_probability=growth_prob,
        fall_probability=fall_prob,
        suggestion=suggestion
    )

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Financial Advisor API. Go to /docs for documentation."}
