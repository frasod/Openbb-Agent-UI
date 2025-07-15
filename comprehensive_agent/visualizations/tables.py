from typing import List, Optional, Dict, Any
import json
from openbb_ai import table

async def generate_tables(widget_data: List[Any]) -> List[Any]:
    tables = []
    
    if not widget_data:
        return tables
    
    for result in widget_data:
        if hasattr(result, 'items'):
            # DataContent object
            for item in result.items:
                try:
                    content = getattr(item, 'content', '')
                    if content:
                        data = parse_table_data(content)
                        if data and len(data) > 0:
                            table_obj = await create_table(data)
                            if table_obj:
                                tables.append(table_obj)
                except Exception as e:
                    print(f"Error generating table: {e}")
                    continue
        elif isinstance(result, dict):
            # Dictionary format (legacy)
            for item in result.get("items", []):
                try:
                    content = item.get("content", "")
                    if content:
                        data = parse_table_data(content)
                        if data and len(data) > 0:
                            table_obj = await create_table(data)
                            if table_obj:
                                tables.append(table_obj)
                except Exception as e:
                    print(f"Error generating table: {e}")
                    continue
    
    return tables

def parse_table_data(content: str) -> Optional[List[Dict]]:
    try:
        content = content.strip()
        if content.startswith('['):
            data = json.loads(content)
            if isinstance(data, list) and len(data) > 0:
                if all(isinstance(item, dict) for item in data):
                    return data
        elif content.startswith('{'):
            data = json.loads(content)
            if isinstance(data, dict):
                return [data]
        return None
    except (json.JSONDecodeError, ValueError):
        return None

async def create_table(data: List[Dict]) -> Optional[Any]:
    try:
        if not data or not isinstance(data, list):
            return None
        
        sample = data[0] if data else {}
        if not isinstance(sample, dict):
            return None
        
        table_name = generate_table_name(data)
        table_description = generate_table_description(data)
        
        return table(
            data=data,
            name=table_name,
            description=table_description
        )
    
    except Exception as e:
        print(f"Error creating table: {e}")
        return None

def generate_table_name(data: List[Dict]) -> str:
    if not data:
        return "Data Table"
    
    sample = data[0]
    keys = list(sample.keys())
    
    if len(keys) == 0:
        return "Data Table"
    elif len(keys) <= 3:
        return f"Table: {', '.join(keys)}"
    else:
        return f"Table: {', '.join(keys[:3])} and {len(keys) - 3} more columns"

def generate_table_description(data: List[Dict]) -> str:
    if not data:
        return "Empty data table"
    
    row_count = len(data)
    sample = data[0]
    column_count = len(sample.keys()) if isinstance(sample, dict) else 0
    
    columns = list(sample.keys()) if isinstance(sample, dict) else []
    
    description = f"Data table with {row_count} rows and {column_count} columns"
    
    if columns:
        if column_count <= 5:
            description += f". Columns: {', '.join(columns)}"
        else:
            description += f". Key columns: {', '.join(columns[:5])}"
    
    return description

def format_table_data(data: List[Dict]) -> List[Dict]:
    if not data:
        return []
    
    formatted_data = []
    for row in data:
        if isinstance(row, dict):
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float) and value.is_integer():
                        formatted_row[key] = int(value)
                    else:
                        formatted_row[key] = round(value, 4) if isinstance(value, float) else value
                elif isinstance(value, str):
                    formatted_row[key] = value.strip()
                else:
                    formatted_row[key] = str(value)
            formatted_data.append(formatted_row)
    
    return formatted_data

def create_sample_table() -> Optional[Any]:
    sample_data = [
        {"Symbol": "AAPL", "Price": 150.25, "Change": 2.5, "Volume": 1000000},
        {"Symbol": "GOOGL", "Price": 2500.00, "Change": -15.0, "Volume": 800000},
        {"Symbol": "MSFT", "Price": 300.75, "Change": 5.25, "Volume": 1200000},
        {"Symbol": "TSLA", "Price": 800.50, "Change": -10.5, "Volume": 900000},
    ]
    
    try:
        return table(
            data=sample_data,
            name="Stock Performance Table",
            description="Table showing stock symbols with price, change, and volume data"
        )
    except Exception as e:
        print(f"Error creating sample table: {e}")
        return None