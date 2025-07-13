import sqlite3

def setup_database():
    """Creates the SQLite database and necessary tables."""
    conn = sqlite3.connect('financial_data.db')
    cursor = conn.cursor()

    # Create stock_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(symbol, date)
        )
    ''')

    # Create news_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            publish_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            sentiment_score REAL
        )
    ''')

    conn.commit()
    conn.close()
    print("Database `financial_data.db` and tables created successfully.")

if __name__ == '__main__':
    setup_database()
