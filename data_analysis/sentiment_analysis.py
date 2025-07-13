from transformers import pipeline
import pandas as pd

# Load a pre-trained model for sentiment analysis in Persian
# Note: This model might need to be downloaded on the first run.
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="HooshvareLab/bert-fa-base-uncased-sentiment-deepsentihugs"
)

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of a given Persian text.

    Args:
        text (str): The input text in Persian.

    Returns:
        dict: A dictionary containing the label ('positive', 'negative', 'neutral')
              and the sentiment score.
    """
    try:
        result = sentiment_pipeline(text)
        return result[0]
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {"label": "neutral", "score": 0.0}

if __name__ == '__main__':
    # Example Usage
    news_headlines = [
        "شاخص کل بورس امروز با رشد چشمگیری همراه بود.",
        "قیمت دلار در بازار آزاد کاهش یافت.",
        "پیش‌بینی‌ها از آینده بازار مسکن نگران‌کننده است.",
        "شرکت خودرو سازی سایپا زیان انباشته خود را افزایش داد."
    ]

    results = [analyze_sentiment(headline) for headline in news_headlines]

    # Create a DataFrame to display results
    df = pd.DataFrame({
        'headline': news_headlines,
        'sentiment_label': [res['label'] for res in results],
        'sentiment_score': [res['score'] for res in results]
    })

    print("Sentiment Analysis Results:")
    print(df)
