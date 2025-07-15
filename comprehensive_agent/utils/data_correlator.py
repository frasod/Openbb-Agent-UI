from typing import List, Dict, Any
import pandas as pd

def correlate_sentiment_with_prices(search_results: List[Dict], openbb_data: pd.DataFrame) -> Dict[str, Any]:
    if not search_results or openbb_data.empty:
        return {'correlation_score': 0.0, 'insights': 'No data available for correlation'}
    avg_sentiment = sum(r['sentiment_score'] for r in search_results) / len(search_results)
    recent_price_change = (openbb_data['close'].iloc[-1] - openbb_data['close'].iloc[0]) / openbb_data['close'].iloc[0]
    correlation_score = avg_sentiment * recent_price_change  # Simple proxy correlation
    insights = f'Average sentiment: {avg_sentiment:.2f}, Price change: {recent_price_change:.2%}'
    return {'correlation_score': correlation_score, 'insights': insights} 