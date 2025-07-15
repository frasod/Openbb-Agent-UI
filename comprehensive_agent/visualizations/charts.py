from typing import List, Optional, Dict, Any
import json
from openbb_ai import chart
from ..processors.error_handler import ErrorHandler, VisualizationError, error_boundary
from ..processors.data_validator import DataValidator
import plotly.graph_objects as go

async def generate_charts(widget_data: List[Any]) -> List[Any]:
    charts = []
    
    if not widget_data:
        return charts
    
    for result in widget_data:
        if hasattr(result, 'items'):
            # DataContent object
            for item in result.items:
                try:
                    content = getattr(item, 'content', '')
                    if content:
                        data = parse_chart_data(content)
                        if data and len(data) > 1:
                            chart_configs = detect_chart_types(data)
                            for config in chart_configs:
                                chart_obj = await create_chart(data, config)
                                if chart_obj:
                                    charts.append(chart_obj)
                except Exception as e:
                    print(f"Error generating chart: {e}")
                    continue
        elif isinstance(result, dict):
            # Dictionary format (legacy)
            for item in result.get("items", []):
                try:
                    content = item.get("content", "")
                    if content:
                        data = parse_chart_data(content)
                        if data and len(data) > 1:
                            chart_configs = detect_chart_types(data)
                            for config in chart_configs:
                                chart_obj = await create_chart(data, config)
                                if chart_obj:
                                    charts.append(chart_obj)
                except Exception as e:
                    print(f"Error generating chart: {e}")
                    continue
    
    return charts

def parse_chart_data(content: str) -> Optional[List[Dict]]:
    try:
        content = content.strip()
        if content.startswith('[') or content.startswith('{'):
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        return None
    except (json.JSONDecodeError, ValueError):
        return None

def detect_chart_types(data: List[Dict]) -> List[Dict]:
    if not data or not isinstance(data, list):
        return []
    
    configs = []
    sample = data[0] if data else {}
    keys = list(sample.keys()) if isinstance(sample, dict) else []
    
    if not keys:
        return []
    
    numeric_keys = []
    date_keys = []
    categorical_keys = []
    
    for key in keys:
        sample_value = sample.get(key)
        if isinstance(sample_value, (int, float)):
            numeric_keys.append(key)
        elif isinstance(sample_value, str):
            if any(date_indicator in key.lower() for date_indicator in ['date', 'time', 'timestamp']):
                date_keys.append(key)
            else:
                categorical_keys.append(key)
    
    if date_keys and numeric_keys:
        configs.append({
            "type": "line",
            "x_key": date_keys[0],
            "y_keys": numeric_keys[:3],
            "name": f"Time Series - {', '.join(numeric_keys[:3])}",
            "description": f"Time series chart showing {', '.join(numeric_keys[:3])} over time"
        })
    
    if categorical_keys and numeric_keys:
        configs.append({
            "type": "bar",
            "x_key": categorical_keys[0],
            "y_keys": numeric_keys[:1],
            "name": f"Bar Chart - {categorical_keys[0]} vs {numeric_keys[0]}",
            "description": f"Bar chart comparing {numeric_keys[0]} across {categorical_keys[0]}"
        })
    
    if len(numeric_keys) >= 2:
        configs.append({
            "type": "scatter",
            "x_key": numeric_keys[0],
            "y_keys": [numeric_keys[1]],
            "name": f"Scatter Plot - {numeric_keys[0]} vs {numeric_keys[1]}",
            "description": f"Scatter plot showing relationship between {numeric_keys[0]} and {numeric_keys[1]}"
        })
    
    if categorical_keys and numeric_keys and len(data) <= 20:
        configs.append({
            "type": "pie",
            "angle_key": numeric_keys[0],
            "callout_label_key": categorical_keys[0],
            "name": f"Pie Chart - {categorical_keys[0]} Distribution",
            "description": f"Pie chart showing distribution of {numeric_keys[0]} by {categorical_keys[0]}"
        })
    
    return configs[:2]

@error_boundary(VisualizationError)
async def create_chart(data: List[Dict], config: Dict) -> Optional[Any]:
    try:
        is_valid, message = DataValidator.validate_chart_data(data)
        if not is_valid:
            await ErrorHandler.log_error_with_context(
                VisualizationError(f"Chart data validation failed: {message}"), 
                data
            )
            return None
        
        chart_kwargs = {
            "data": data,
            "name": config["name"],
            "description": config["description"]
        }
        
        if config["type"] in ["line", "bar", "scatter"]:
            chart_kwargs["x_key"] = config["x_key"]
            chart_kwargs["y_keys"] = config["y_keys"]
        elif config["type"] in ["pie", "donut"]:
            chart_kwargs["angle_key"] = config["angle_key"]
            chart_kwargs["callout_label_key"] = config["callout_label_key"]
        
        return chart(config["type"], **chart_kwargs)
    
    except Exception as e:
        error_info = await ErrorHandler.handle_data_error(e, {"chart_type": config.get("type", "unknown")})
        print(f"Chart creation failed: {error_info['message']}")
        return None

def create_sample_charts() -> List[Any]:
    sample_data = [
        {"date": "2024-01", "revenue": 1000, "profit": 200, "category": "Q1"},
        {"date": "2024-02", "revenue": 1200, "profit": 250, "category": "Q1"},
        {"date": "2024-03", "revenue": 1100, "profit": 220, "category": "Q1"},
        {"date": "2024-04", "revenue": 1300, "profit": 280, "category": "Q2"},
    ]
    
    charts = []
    
    try:
        line_chart = chart(
            "line",
            data=sample_data,
            x_key="date",
            y_keys=["revenue", "profit"],
            name="Revenue and Profit Trend",
            description="Line chart showing revenue and profit over time"
        )
        charts.append(line_chart)
        
        pie_chart = chart(
            "pie",
            data=sample_data,
            angle_key="revenue",
            callout_label_key="category",
            name="Revenue Distribution",
            description="Pie chart showing revenue distribution by quarter"
        )
        charts.append(pie_chart)
        
    except Exception as e:
        print(f"Error creating sample charts: {e}")
    
    return charts

def generate_sentiment_trend_chart(search_results: List[Dict]) -> go.Figure:
    if not search_results:
        return go.Figure()
    sentiments = [r['sentiment_score'] for r in search_results]
    titles = [r['title'][:30] for r in search_results]
    fig = go.Figure([go.Bar(x=titles, y=sentiments, name='Sentiment')])
    fig.update_layout(title='News Sentiment Trends', xaxis_title='Article', yaxis_title='Sentiment Score')
    return fig