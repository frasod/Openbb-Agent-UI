from typing import Dict, List, Any, Optional, Union
import json
import logging
from datetime import datetime
from .error_handler import ErrorHandler, DataProcessingError, error_boundary
from .data_validator import DataValidator

logger = logging.getLogger(__name__)

@error_boundary(DataProcessingError)
async def process_api_data(widget_data: Union[List[Any], Any]) -> Optional[Dict[str, Any]]:
    """Process JSON/API data from OpenBB widgets with advanced structuring.
    
    Returns enriched data with metadata, validation, and analysis.
    """
    if not widget_data:
        return None
    
    try:
        processed_data = {
            "datasets": [],
            "summary": {"total_records": 0, "data_types": {}, "quality_score": 0.0},
            "metadata": {"processed_at": datetime.now().isoformat()},
            "analysis": {}
        }
        
        if isinstance(widget_data, list):
            for result in widget_data:
                dataset = await _process_single_api_source(result)
                if dataset:
                    processed_data["datasets"].append(dataset)
        else:
            dataset = await _process_single_api_source(widget_data)
            if dataset:
                processed_data["datasets"].append(dataset)
        
        if processed_data["datasets"]:
            _compute_summary_metrics(processed_data)
            processed_data["analysis"] = _perform_data_analysis(processed_data["datasets"])
            return processed_data
            
        return None
        
    except Exception as e:
        await ErrorHandler.log_error_with_context(
            DataProcessingError(f"API data processing failed: {e}"), 
            widget_data
        )
        return None

async def _process_single_api_source(data_source: Any) -> Optional[Dict[str, Any]]:
    """Process a single API data source"""
    try:
        if hasattr(data_source, 'items'):
            return await _process_datacontent_api(data_source)
        elif isinstance(data_source, dict):
            return await _process_dict_api(data_source)
        return None
    except Exception as e:
        logger.error(f"Error processing API source: {e}")
        return None

async def _process_datacontent_api(data_source: Any) -> Optional[Dict[str, Any]]:
    """Process DataContent format API data"""
    dataset = {
        "source_type": "datacontent",
        "records": [],
        "schema": {},
        "metadata": {}
    }
    
    for item in data_source.items:
        if _is_json_item(item):
            json_data = await _extract_json_from_item(item)
            if json_data:
                dataset["records"].extend(_normalize_json_data(json_data))
    
    if dataset["records"]:
        dataset["schema"] = _infer_schema(dataset["records"])
        dataset["metadata"] = _extract_metadata(data_source)
        return dataset
        
    return None

async def _process_dict_api(data_source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process dictionary format API data"""
    dataset = {
        "source_type": "dictionary",
        "records": [],
        "schema": {},
        "metadata": {}
    }
    
    if "items" in data_source:
        for item in data_source["items"]:
            if item.get("format") == "JsonDataFormat":
                json_data = await _extract_json_content(item)
                if json_data:
                    dataset["records"].extend(_normalize_json_data(json_data))
    else:
        dataset["records"] = _normalize_json_data(data_source)
    
    if dataset["records"]:
        dataset["schema"] = _infer_schema(dataset["records"])
        return dataset
        
    return None

def _is_json_item(item: Any) -> bool:
    """Check if an item contains JSON data"""
    try:
        data_format_name = str(type(getattr(item, 'data_format', None)).__name__)
        return data_format_name in ['JsonDataFormat', 'ApiDataFormat']
    except:
        return False

async def _extract_json_from_item(item: Any) -> Optional[Any]:
    """Extract JSON data from OpenBB data item"""
    try:
        if hasattr(item, 'url'):
            return await _fetch_json_from_url(str(item.url))
        
        content = getattr(item, 'content', '')
        if content:
            if isinstance(content, str):
                return json.loads(content)
            return content
            
    except Exception as e:
        logger.error(f"JSON extraction from item failed: {e}")
        return None
    
    return None

async def _fetch_json_from_url(url: str) -> Optional[Any]:
    """Fetch JSON data from URL"""
    try:
        import httpx
        
        logger.info(f"Fetching JSON from {url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Failed to fetch JSON from {url}: {e}")
        return None

def _normalize_json_data(data: Any) -> List[Dict[str, Any]]:
    """Normalize JSON data to list of dictionaries"""
    if isinstance(data, list):
        normalized = []
        for item in data:
            if isinstance(item, dict):
                normalized.append(_flatten_nested_dict(item))
            else:
                normalized.append({"value": item, "type": type(item).__name__})
        return normalized
    elif isinstance(data, dict):
        return [_flatten_nested_dict(data)]
    else:
        return [{"value": data, "type": type(data).__name__}]

def _flatten_nested_dict(data: Dict[str, Any], prefix: str = "", max_depth: int = 3) -> Dict[str, Any]:
    """Flatten nested dictionary with depth limit"""
    if max_depth <= 0:
        return {prefix.rstrip("_"): str(data)}
    
    flattened = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict) and len(value) > 0:
            flattened.update(_flatten_nested_dict(value, f"{new_key}_", max_depth - 1))
        elif isinstance(value, list) and len(value) > 0:
            if all(isinstance(item, dict) for item in value):
                for i, item in enumerate(value[:5]):
                    flattened.update(_flatten_nested_dict(item, f"{new_key}_{i}_", max_depth - 1))
            else:
                flattened[new_key] = _serialize_list_value(value)
        else:
            flattened[new_key] = _serialize_value(value)
    
    return flattened

def _serialize_value(value: Any) -> Any:
    """Serialize individual values for JSON compatibility"""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, datetime):
        return value.isoformat()
    else:
        return str(value)

def _serialize_list_value(value: List[Any]) -> str:
    """Serialize list values to string representation"""
    if len(value) <= 10:
        return str(value)
    else:
        return f"[{', '.join(str(v) for v in value[:5])}, ... +{len(value)-5} more]"

def _infer_schema(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Infer schema from record data"""
    schema = {
        "fields": {},
        "total_fields": 0,
        "nullable_fields": [],
        "numeric_fields": [],
        "date_fields": []
    }
    
    if not records:
        return schema
    
    all_keys = set()
    for record in records:
        all_keys.update(record.keys())
    
    schema["total_fields"] = len(all_keys)
    
    for key in all_keys:
        values = [record.get(key) for record in records]
        non_null_values = [v for v in values if v is not None]
        
        field_info = {
            "type": _infer_field_type(non_null_values),
            "nullable": len(non_null_values) < len(values),
            "unique_values": len(set(str(v) for v in non_null_values)),
            "sample_values": list(set(str(v) for v in non_null_values))[:5]
        }
        
        schema["fields"][key] = field_info
        
        if field_info["nullable"]:
            schema["nullable_fields"].append(key)
        
        if field_info["type"] in ["integer", "float"]:
            schema["numeric_fields"].append(key)
        elif field_info["type"] == "datetime":
            schema["date_fields"].append(key)
    
    return schema

def _infer_field_type(values: List[Any]) -> str:
    """Infer the most likely type for a field"""
    if not values:
        return "unknown"
    
    type_counts = {}
    for value in values:
        inferred_type = _get_value_type(value)
        type_counts[inferred_type] = type_counts.get(inferred_type, 0) + 1
    
    return max(type_counts, key=type_counts.get)

def _get_value_type(value: Any) -> str:
    """Get the type of a single value"""
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        if _is_date_string(value):
            return "datetime"
        elif _is_numeric_string(value):
            return "numeric_string"
        return "string"
    else:
        return "object"

def _is_date_string(value: str) -> bool:
    """Check if string represents a date"""
    import re
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}',
        r'^\d{2}/\d{2}/\d{4}',
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
    ]
    return any(re.match(pattern, value) for pattern in date_patterns)

def _is_numeric_string(value: str) -> bool:
    """Check if string represents a number"""
    try:
        float(value.replace(',', ''))
        return True
    except ValueError:
        return False

def _extract_metadata(data_source: Any) -> Dict[str, Any]:
    """Extract metadata from data source"""
    metadata = {"extraction_method": "datacontent"}
    
    if hasattr(data_source, 'items') and data_source.items:
        first_item = data_source.items[0]
        if hasattr(first_item, 'data_format'):
            metadata["format_type"] = str(type(first_item.data_format).__name__)
    
    return metadata

def _compute_summary_metrics(processed_data: Dict[str, Any]) -> None:
    """Compute summary metrics for all datasets"""
    summary = processed_data["summary"]
    datasets = processed_data["datasets"]
    
    total_records = sum(len(ds["records"]) for ds in datasets)
    summary["total_records"] = total_records
    
    all_types = set()
    quality_scores = []
    
    for dataset in datasets:
        schema = dataset["schema"]
        
        for field_info in schema["fields"].values():
            all_types.add(field_info["type"])
        
        nullable_ratio = len(schema["nullable_fields"]) / max(1, schema["total_fields"])
        quality_score = 1.0 - (nullable_ratio * 0.5)
        quality_scores.append(quality_score)
    
    summary["data_types"] = list(all_types)
    summary["quality_score"] = sum(quality_scores) / max(1, len(quality_scores))

def _perform_data_analysis(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform advanced analysis on datasets"""
    analysis = {
        "field_distribution": {},
        "data_patterns": [],
        "recommendations": []
    }
    
    all_fields = {}
    
    for dataset in datasets:
        for field_name, field_info in dataset["schema"]["fields"].items():
            if field_name not in all_fields:
                all_fields[field_name] = []
            all_fields[field_name].append(field_info["type"])
    
    for field_name, types in all_fields.items():
        type_consistency = len(set(types)) == 1
        analysis["field_distribution"][field_name] = {
            "appears_in_datasets": len(types),
            "type_consistent": type_consistency,
            "dominant_type": max(set(types), key=types.count)
        }
        
        if not type_consistency:
            analysis["data_patterns"].append(f"Field '{field_name}' has inconsistent types across datasets")
    
    if len(datasets) > 1:
        analysis["recommendations"].append("Consider data schema alignment across datasets")
    
    numeric_fields = sum(len(ds["schema"]["numeric_fields"]) for ds in datasets)
    if numeric_fields > 0:
        analysis["recommendations"].append("Datasets contain numeric fields suitable for quantitative analysis")
    
    return analysis

async def _extract_json_content(json_data: Dict[str, Any]) -> Optional[Any]:
    """Legacy method for dictionary format JSON extraction"""
    try:
        content = json_data.get("content", {})
        if isinstance(content, dict):
            if "url" in content:
                return await _fetch_json_from_url(content["url"])
            elif "data" in content:
                return json.loads(content["data"]) if isinstance(content["data"], str) else content["data"]
        return content
    except Exception as e:
        logger.error(f"Error extracting JSON content: {e}")
        return None

async def prepare_api_data_for_visualization(dataset: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Prepare API data for visualization"""
    try:
        records = dataset.get("records", [])
        if not records:
            return None
        
        schema = dataset.get("schema", {})
        numeric_fields = schema.get("numeric_fields", [])
        
        if not numeric_fields:
            logger.warning("No numeric fields found for visualization")
            return None
        
        prepared_data = []
        for record in records:
            viz_record = {}
            for key, value in record.items():
                if key in numeric_fields:
                    viz_record[key] = DataValidator._clean_numeric_value(value)
                else:
                    viz_record[key] = str(value) if value is not None else ""
            prepared_data.append(viz_record)
        
        is_valid, message = DataValidator.validate_chart_data(prepared_data)
        if not is_valid:
            logger.warning(f"API data not suitable for visualization: {message}")
            return None
        
        return prepared_data
        
    except Exception as e:
        logger.error(f"Failed to prepare API data for visualization: {e}")
        return None