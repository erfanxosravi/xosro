import pandas as pd

def calculate_moving_average(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculates the simple moving average (SMA)."""
    return data['close'].rolling(window=window).mean()

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()

    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()

    return macd, signal_line

if __name__ == '__main__':
    # Create a sample dataframe for testing
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'close': [100, 102, 101, 103, 105]
    }
    df = pd.DataFrame(data).set_index('date')

    # --- Test Calculations ---
    df['sma_20'] = calculate_moving_average(df, window=3)
    df['rsi_14'] = calculate_rsi(df, window=3)
    df['macd'], df['macd_signal'] = calculate_macd(df, fast_period=2, slow_period=4, signal_period=2)

    print("Technical Indicators Calculation Results:")
    print(df)
