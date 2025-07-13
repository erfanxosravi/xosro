import requests
import pandas as pd
from datetime import datetime

def get_stock_history(symbol: str, days: int = 365):
    """
    Fetches historical stock data from tsetmc.com for a given symbol.

    Args:
        symbol (str): The stock symbol (e.g., 'خودرو').
        days (int): The number of past days to fetch data for.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the historical data.
    """
    url = f"http://www.tsetmc.com/tsev2/data/Export-All.aspx?t=i&a=1&f=0&e=0&d=0&s={symbol}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # The data is returned as CSV, so we can use pandas to read it
        df = pd.read_csv(response.text)

        # Data cleaning and formatting
        df.columns = df.columns.str.strip().str.lower().str.replace('[<>]', '', regex=True)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.set_index('date')

        # Filter data for the last 'days'
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=days)
        df = df[df.index >= start_date]

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    symbol = "323"  # Symbol for فولاد مبارکه
    stock_data = get_stock_history(symbol, days=90)

    if stock_data is not None:
        print(stock_data.head())
        # Save to CSV
        stock_data.to_csv('folad_mobarakeh.csv')
        print("Data saved to folad_mobarakeh.csv")
