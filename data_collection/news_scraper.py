import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_eghtesadnews():
    """Scrapes news from eghtesadnews.com"""
    url = "https://www.eghtesadnews.com/service/archive"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        titles = [h2.get_text(strip=True) for h2 in soup.find_all('h2', class_='title')]
        return titles
    except requests.exceptions.RequestException as e:
        print(f"Error scraping Eghtesadnews: {e}")
        return []

def scrape_irna():
    """Scrapes news from irna.ir"""
    url = "https://www.irna.ir/service/archive"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        titles = [h2.get_text(strip=True) for h2 in soup.find_all('h2', class_='title')]
        return titles
    except requests.exceptions.RequestException as e:
        print(f"Error scraping IRNA: {e}")
        return []


if __name__ == '__main__':
    print("Scraping Eghtesadnews...")
    eghtesad_titles = scrape_eghtesadnews()
    print(f"Found {len(eghtesad_titles)} titles.")

    print("\nScraping IRNA...")
    irna_titles = scrape_irna()
    print(f"Found {len(irna_titles)} titles.")

    # Create a DataFrame and save to CSV
    all_titles = eghtesad_titles + irna_titles
    df = pd.DataFrame({'source': ['Eghtesadnews'] * len(eghtesad_titles) + ['IRNA'] * len(irna_titles),
                       'title': all_titles})
    df.to_csv('news_headlines.csv', index=False)
    print("\nNews headlines saved to news_headlines.csv")
