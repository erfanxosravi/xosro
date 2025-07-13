import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from hybrid_model import HybridModel

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]
        y = data[i+seq_length, -1] # The last column is the target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == '__main__':
    # --- 1. Load and Prepare Dummy Data ---
    # Imagine this data comes from our database
    data = {
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 105,
        'low': np.random.rand(100) * 95,
        'close': np.random.rand(100) * 102,
        'volume': np.random.randint(10000, 50000, 100),
        'sentiment_score': np.random.rand(100) * 2 - 1, # a score between -1 and 1
        'target': np.random.randint(0, 2, 100) # 0 for fall, 1 for growth
    }
    df = pd.DataFrame(data)

    # Normalize features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df.drop('target', axis=1))

    # Combine features and target
    scaled_data = np.c_[scaled_features, df['target']]

    # --- 2. Create Sequences ---
    sequence_length = 10
    X, y = create_sequences(scaled_data, sequence_length)

    # Split sentiment scores from the main data
    X_ts = X[:, :, :-1] # Time-series features
    X_sentiment = X[:, -1, -1].reshape(-1, 1) # Sentiment score from the last day of the sequence

    # Convert to PyTorch tensors
    X_ts_tensor = torch.from_numpy(X_ts).float()
    X_sentiment_tensor = torch.from_numpy(X_sentiment).float()
    y_tensor = torch.from_numpy(y).long()

    # --- 3. Model, Loss, and Optimizer ---
    model = HybridModel(
        input_size=X_ts.shape[2],
        lstm_hidden_size=50,
        lstm_num_layers=2,
        mlp_hidden_size=25,
        output_size=2
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 4. Training Loop ---
    epochs = 20
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_ts_tensor, X_sentiment_tensor)

        # Calculate loss
        loss = criterion(outputs, y_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # --- 5. Evaluation (on the same data for simplicity) ---
    model.eval()
    with torch.no_grad():
        outputs = model(X_ts_tensor, X_sentiment_tensor)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_tensor).sum().item()
        accuracy = 100 * correct / len(y_tensor)
        print(f'\nTraining Accuracy: {accuracy:.2f}%')
