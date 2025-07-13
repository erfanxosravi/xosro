import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class HybridModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, mlp_hidden_size, output_size):
        super(HybridModel, self).__init__()

        # LSTM layer for time-series data
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)

        # MLP for combining LSTM output with sentiment score
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size + 1, mlp_hidden_size), # +1 for sentiment score
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, time_series_data, sentiment_score):
        # LSTM forward pass
        lstm_out, _ = self.lstm(time_series_data)

        # Get the last hidden state of the LSTM
        lstm_last_hidden_state = lstm_out[:, -1, :]

        # Concatenate LSTM output with sentiment score
        combined_input = torch.cat((lstm_last_hidden_state, sentiment_score), dim=1)

        # MLP forward pass
        output = self.mlp(combined_input)
        return output

if __name__ == '__main__':
    # --- Dummy Data for Testing ---
    # Parameters
    input_size = 5  # (open, high, low, close, volume)
    sequence_length = 10 # 10 days of historical data
    lstm_hidden_size = 50
    lstm_num_layers = 2
    mlp_hidden_size = 25
    output_size = 2  # (Prob_growth, Prob_fall)
    batch_size = 32

    # Model Initialization
    model = HybridModel(input_size, lstm_hidden_size, lstm_num_layers, mlp_hidden_size, output_size)
    print("Model Architecture:")
    print(model)

    # Dummy Input
    time_series_input = torch.randn(batch_size, sequence_length, input_size)
    sentiment_input = torch.randn(batch_size, 1) # Single sentiment score per sequence

    # Forward pass
    predictions = model(time_series_input, sentiment_input)

    print("\nOutput shape:", predictions.shape)
    print("Sample prediction:", predictions[0].tolist())
    # The output should be probabilities that sum to 1
    print("Sum of probabilities:", torch.sum(predictions[0]).item())
