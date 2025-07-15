from typing import List, Dict
from duckduckgo_search import DDGS
from textblob import TextBlob
import datetime as dt

class FinancialWebSearcher:
    def __init__(self, source_whitelist: List[str] = ["bloomberg.com", "reuters.com"], max_age_days: int = 7):
        self.ddgs = DDGS()
        self.source_whitelist = source_whitelist
        self.max_age = dt.timedelta(days=max_age_days)

    def search_and_analyze(self, query: str, max_results: int = 10) -> List[Dict]:
        results = self.ddgs.text(query, max_results=max_results * 2)  # Fetch more to filter
        now = dt.datetime.now()
        filtered_results = [
            r for r in results
            if any(s in r["href"] for s in self.source_whitelist)
            and ('date' not in r or now - dt.datetime.fromisoformat(r['date']) <= self.max_age)
        ][:max_results]
        return [
            {
                "title": r["title"],
                "snippet": r["body"],
                "url": r["href"],
                "sentiment_score": TextBlob(r["body"]).sentiment.polarity
            }
            for r in filtered_results
        ] 