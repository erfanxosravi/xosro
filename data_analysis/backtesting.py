import pandas as pd
import numpy as np
import torch
from modeling.hybrid_model import HybridModel # Adjust import path as needed
from data_analysis.technical_analysis import calculate_moving_average, calculate_rsi

class Backtester:
    def __init__(self, model, data, initial_capital=100000.0):
        self.model = model
        self.data = data
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.shares = 0
        self.portfolio_values = []

    def _get_signal(self, historical_data, sentiment_score):
        """Generates a trading signal using the model."""
        self.model.eval()
        with torch.no_grad():
            time_series_tensor = torch.from_numpy(np.array([historical_data])).float()
            sentiment_tensor = torch.from_numpy(np.array([[sentiment_score]])).float()

            prediction = self.model(time_series_tensor, sentiment_tensor)
            probabilities = prediction[0].tolist()

            if probabilities[1] > 0.7: return "Buy"
            if probabilities[0] > 0.7: return "Sell"
            return "Hold"

    def run(self, sequence_length=10):
        """Runs the backtesting simulation."""
        for i in range(sequence_length, len(self.data)):
            current_price = self.data.iloc[i]['close']

            # Prepare data for the model
            historical_data = self.data.iloc[i-sequence_length:i][['open', 'high', 'low', 'close', 'volume']].values
            sentiment_score = self.data.iloc[i-1]['sentiment_score'] # Use previous day's sentiment

            # Get signal
            signal = self._get_signal(historical_data, sentiment_score)

            # Execute trade based on signal
            if signal == "Buy" and self.capital > current_price:
                shares_to_buy = self.capital // current_price
                self.shares += shares_to_buy
                self.capital -= shares_to_buy * current_price
                # print(f"Date: {self.data.index[i].date()}, Action: Buy, Price: {current_price:.2f}")

            elif signal == "Sell" and self.shares > 0:
                self.capital += self.shares * current_price
                self.shares = 0
                # print(f"Date: {self.data.index[i].date()}, Action: Sell, Price: {current_price:.2f}")

            # Update portfolio value
            portfolio_value = self.capital + self.shares * current_price
            self.portfolio_values.append(portfolio_value)

        return self.portfolio_values

def calculate_performance(portfolio_values, initial_capital):
    """Calculates and prints performance metrics."""
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) # Annualized Sharpe Ratio

    print("\n--- Backtesting Performance ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("-----------------------------")


if __name__ == '__main__':
    # --- Dummy Data for Backtesting ---
    data = {
        'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100)),
        'open': np.random.uniform(98, 102, 100),
        'high': np.random.uniform(100, 105, 100),
        'low': np.random.uniform(95, 99, 100),
        'close': np.random.uniform(99, 103, 100),
        'volume': np.random.randint(10000, 50000, 100),
        'sentiment_score': np.random.uniform(-1, 1, 100)
    }
    df = pd.DataFrame(data).set_index('date')

    # --- Load Model ---
    # In a real scenario, you'd load your trained model
    model = HybridModel(input_size=5, lstm_hidden_size=50, lstm_num_layers=2, mlp_hidden_size=25, output_size=2)

    # --- Run Backtester ---
    backtester = Backtester(model, df, initial_capital=100000.0)
    portfolio_values = backtester.run(sequence_length=10)

    # --- Calculate Performance ---
    if portfolio_values:
        calculate_performance(portfolio_values, backtester.initial_capital)
    else:
        print("Backtesting did not produce any results.")
