"""
This module defines the tools available to the OpenBB Agent.
It follows a structured, function-first approach to tool definition,
ensuring clarity and ease of extension.
"""
from openbb import obb
import json
from ..visualizations.financial_charts import candlestick_chart
import asyncio

def get_news(symbol: str) -> str:
    """
    Get the latest financial news for a specific stock symbol.
    Returns a JSON string of the news data.
    """
    try:
        print(f"🗞️ Getting news for {symbol}...")
        # Execute the OpenBB function and convert the DataFrame to JSON
        news_data = obb.news.company(symbol=symbol, limit=5).to_df()
        print(f"✅ Retrieved {len(news_data)} news articles for {symbol}")
        return news_data.to_json(orient="records")
    except Exception as e:
        error_msg = f"Error fetching news for {symbol}: {e}"
        print(f"❌ {error_msg}")
        return error_msg

def get_historical_price(symbol: str) -> str:
    """
    Get historical daily price data for a specific stock symbol.
    Returns a JSON string of the historical price data.
    """
    try:
        print(f"📈 Getting historical price data for {symbol}...")
        # Execute the OpenBB function and convert the DataFrame to JSON
        price_data = obb.equity.price.historical(symbol=symbol, interval="1d", provider="fmp").to_df()
        print(f"✅ Retrieved {len(price_data)} price data points for {symbol}")
        return price_data.to_json(orient="records")
    except Exception as e:
        error_msg = f"Error fetching historical price for {symbol}: {e}"
        print(f"❌ {error_msg}")
        return error_msg

async def generate_financial_chart(symbol: str) -> str:
    """
    Generates a candlestick chart for the historical daily price data of a specific stock symbol.
    Returns a JSON string of the chart artifact.
    """
    try:
        print(f"📊 CHART GENERATION STARTED for {symbol}...")
        
        # Try different providers in order of preference (free first)
        providers = ["yfinance", "intrinio", "polygon", "fmp"]
        price_data = None
        
        for provider in providers:
            try:
                print(f"🔄 Trying provider: {provider}")
                price_data = obb.equity.price.historical(
                    symbol=symbol, 
                    interval="1d", 
                    provider=provider,
                    limit=100  # Limit to last 100 days to reduce load
                ).to_df()
                print(f"✅ Success with provider: {provider}")
                break
            except Exception as provider_error:
                print(f"❌ Provider {provider} failed: {provider_error}")
                continue
        
        if price_data is None or price_data.empty:
            # If all providers fail, create sample data for demonstration
            print("🔧 Creating sample chart data for demonstration...")
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Generate sample OHLC data
            dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
            base_price = 150.0
            
            sample_data = []
            for i, date in enumerate(dates):
                daily_change = np.random.normal(0, 2)
                open_price = base_price + daily_change
                high_price = open_price + abs(np.random.normal(0, 3))
                low_price = open_price - abs(np.random.normal(0, 3))
                close_price = open_price + np.random.normal(0, 2)
                base_price = close_price
                
                sample_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": int(np.random.uniform(1000000, 5000000))
                })
            
            records = sample_data
            print(f"✅ Created sample data with {len(records)} points")
        else:
            print(f"✅ Retrieved {len(price_data)} data points for {symbol}")
            print(f"📋 Data columns: {list(price_data.columns)}")
            
            # Convert to records format
            records = price_data.to_dict('records')
            print(f"📝 Sample record: {records[0] if records else 'No data'}")
        
        # Create chart structure
        chart_dict = {
            "type": "candlestick",
            "data": records,
            "x_key": "date",
            "ohlc_keys": {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close"
            },
            "name": f"{symbol} Stock Chart",
            "description": f"Candlestick chart for {symbol}" + (" (Sample Data)" if price_data is None else " (Live Data)")
        }
        
        print(f"✅ CHART CREATED with {len(records)} data points!")
        return json.dumps(chart_dict)
        
    except Exception as e:
        error_msg = f"❌ CHART ERROR for {symbol}: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

# --- Tool Manifest ---
# A clear, organized list of all available tools.
# This makes it easy to see what the agent can do and to add new tools.
AVAILABLE_TOOLS = {
    "get_news": {
        "function": get_news,
        "description": "Get the latest financial news for a specific stock symbol. Use this for text-based news analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol (e.g., 'AAPL', 'TSLA').",
                },
            },
            "required": ["symbol"],
        },
    },
    "get_historical_price": {
        "function": get_historical_price,
        "description": "Get historical daily price data for a specific stock symbol. Use this to get raw price data for analysis. Trigger this tool for queries like 'price history for MSFT'.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol (e.g., 'AAPL', 'TSLA').",
                },
            },
            "required": ["symbol"],
        },
    },
    "generate_financial_chart": {
        "function": generate_financial_chart,
        "description": "Generate a candlestick chart for the historical daily price data of a stock. Trigger this tool for queries like 'show me a chart of AAPL stock'.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol (e.g., 'AAPL', 'TSLA').",
                },
            },
            "required": ["symbol"],
        },
    },
    # Add more tool definitions here, following the same structure.
}

def get_tool_definitions() -> list[dict]:
    """
    Returns a list of tool definitions formatted for the OpenAI API.
    This keeps the API-specific formatting separate from the core tool logic.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": details["description"],
                "parameters": details["parameters"],
            },
        }
        for name, details in AVAILABLE_TOOLS.items()
    ] 