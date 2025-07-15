from typing import List, Dict, Any, Optional, Union
import pandas as pd
import io
import base64
import logging
from pathlib import Path
from .error_handler import ErrorHandler, DataProcessingError, error_boundary
from .data_validator import DataValidator

logger = logging.getLogger(__name__)

@error_boundary(DataProcessingError)
async def process_spreadsheet_data(widget_data: Union[List[Any], Any]) -> Optional[Dict[str, Any]]:
    """Process spreadsheet data (Excel/CSV) from OpenBB widgets.
    
    Returns structured data with metadata and validation results.
    """
    if not widget_data:
        return None
    
    try:
        processed_data = {
            "sheets": [],
            "summary": {"total_rows": 0, "total_columns": 0, "data_quality": {}},
            "metadata": {}
        }
        
        if isinstance(widget_data, list):
            for result in widget_data:
                sheet_data = await _process_single_spreadsheet_source(result)
                if sheet_data:
                    processed_data["sheets"].extend(sheet_data["sheets"])
                    _update_summary(processed_data["summary"], sheet_data)
        else:
            sheet_data = await _process_single_spreadsheet_source(widget_data)
            if sheet_data:
                processed_data["sheets"].extend(sheet_data["sheets"])
                _update_summary(processed_data["summary"], sheet_data)
        
        if processed_data["sheets"]:
            processed_data["summary"]["data_quality"] = _analyze_data_quality(processed_data["sheets"])
            return processed_data
            
        return None
        
    except Exception as e:
        await ErrorHandler.log_error_with_context(
            DataProcessingError(f"Spreadsheet processing failed: {e}"), 
            widget_data
        )
        return None

async def _process_single_spreadsheet_source(data_source: Any) -> Optional[Dict[str, Any]]:
    """Process a single spreadsheet data source"""
    try:
        if hasattr(data_source, 'items'):
            return await _process_datacontent_spreadsheet(data_source)
        elif isinstance(data_source, dict):
            return await _process_dict_spreadsheet(data_source)
        return None
    except Exception as e:
        logger.error(f"Error processing spreadsheet source: {e}")
        return None

async def _process_datacontent_spreadsheet(data_source: Any) -> Optional[Dict[str, Any]]:
    """Process DataContent format spreadsheet"""
    result = {"sheets": [], "metadata": {}}
    
    for item in data_source.items:
        if _is_spreadsheet_item(item):
            sheet_data = await _extract_spreadsheet_from_item(item)
            if sheet_data:
                result["sheets"].append(sheet_data)
    
    return result if result["sheets"] else None

async def _process_dict_spreadsheet(data_source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process dictionary format spreadsheet"""
    result = {"sheets": [], "metadata": {}}
    
    for item in data_source.get("items", []):
        if item.get("format") in ["ExcelDataFormat", "CsvDataFormat"]:
            sheet_data = await _extract_spreadsheet_content(item)
            if sheet_data:
                result["sheets"].append(sheet_data)
    
    return result if result["sheets"] else None

def _is_spreadsheet_item(item: Any) -> bool:
    """Check if an item contains spreadsheet data"""
    try:
        data_format_name = str(type(getattr(item, 'data_format', None)).__name__)
        return data_format_name in ['ExcelDataFormat', 'CsvDataFormat']
    except:
        return False

async def _extract_spreadsheet_from_item(item: Any) -> Optional[Dict[str, Any]]:
    """Extract spreadsheet data from OpenBB data item"""
    try:
        if hasattr(item, 'url'):
            return await _download_and_process_spreadsheet(str(item.url))
        
        content = getattr(item, 'content', '')
        if content:
            filename = getattr(getattr(item, 'data_format', None), 'filename', 'spreadsheet')
            return _process_spreadsheet_content(content, filename)
            
    except Exception as e:
        logger.error(f"Spreadsheet extraction from item failed: {e}")
        return None
    
    return None

async def _download_and_process_spreadsheet(url: str) -> Optional[Dict[str, Any]]:
    """Download spreadsheet from URL and process"""
    try:
        import httpx
        
        logger.info(f"Downloading spreadsheet from {url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            
            filename = Path(url).name
            return _process_spreadsheet_bytes(response.content, filename)
            
    except Exception as e:
        logger.error(f"Failed to download spreadsheet from {url}: {e}")
        return None

def _process_spreadsheet_content(content: str, filename: str) -> Optional[Dict[str, Any]]:
    """Process base64-encoded spreadsheet content"""
    try:
        spreadsheet_bytes = base64.b64decode(content)
        return _process_spreadsheet_bytes(spreadsheet_bytes, filename)
    except Exception as e:
        logger.error(f"Base64 spreadsheet processing failed: {e}")
        return None

def _process_spreadsheet_bytes(data: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Process spreadsheet bytes using pandas"""
    try:
        data_io = io.BytesIO(data)
        file_ext = Path(filename).suffix.lower()
        
        if file_ext in ['.xlsx', '.xls']:
            return _process_excel_data(data_io, filename)
        elif file_ext == '.csv':
            return _process_csv_data(data_io, filename)
        else:
            return _attempt_auto_detection(data_io, filename)
            
    except Exception as e:
        logger.error(f"Spreadsheet bytes processing failed: {e}")
        return None

def _process_excel_data(data_io: io.BytesIO, filename: str) -> Dict[str, Any]:
    """Process Excel file with multiple sheets support"""
    excel_file = pd.ExcelFile(data_io)
    sheets_data = []
    
    for sheet_name in excel_file.sheet_names:
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            if not df.empty:
                sheet_info = {
                    "name": sheet_name,
                    "filename": filename,
                    "data": _prepare_dataframe_for_output(df),
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "summary_stats": _generate_summary_stats(df)
                }
                sheets_data.append(sheet_info)
                
        except Exception as e:
            logger.warning(f"Failed to process sheet '{sheet_name}': {e}")
            continue
    
    return {"sheets": sheets_data, "metadata": {"source_file": filename, "format": "excel"}}

def _process_csv_data(data_io: io.BytesIO, filename: str) -> Dict[str, Any]:
    """Process CSV file with encoding detection"""
    encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            data_io.seek(0)
            df = pd.read_csv(data_io, encoding=encoding)
            
            if not df.empty:
                sheet_info = {
                    "name": "Sheet1",
                    "filename": filename,
                    "data": _prepare_dataframe_for_output(df),
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "summary_stats": _generate_summary_stats(df),
                    "encoding": encoding
                }
                
                return {"sheets": [sheet_info], "metadata": {"source_file": filename, "format": "csv"}}
                
        except Exception as e:
            logger.debug(f"Failed to read CSV with {encoding}: {e}")
            continue
    
    raise ValueError(f"Unable to read CSV file {filename} with any supported encoding")

def _attempt_auto_detection(data_io: io.BytesIO, filename: str) -> Optional[Dict[str, Any]]:
    """Attempt to auto-detect file format"""
    data_io.seek(0)
    sample = data_io.read(1024)
    data_io.seek(0)
    
    if b'PK' in sample[:4]:
        return _process_excel_data(data_io, filename)
    elif b',' in sample or b';' in sample:
        return _process_csv_data(data_io, filename)
    else:
        logger.warning(f"Unable to detect format for file: {filename}")
        return None

def _prepare_dataframe_for_output(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to JSON-serializable format with data cleaning"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
        elif pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean = df_clean.fillna('')
    
    return df_clean.to_dict('records')

def _generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for DataFrame"""
    stats = {}
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats["numeric"] = df[numeric_cols].describe().to_dict()
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        stats["categorical"] = {}
        for col in categorical_cols:
            stats["categorical"][col] = {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
    
    return stats

def _update_summary(summary: Dict[str, Any], sheet_data: Dict[str, Any]) -> None:
    """Update overall summary with sheet data"""
    for sheet in sheet_data.get("sheets", []):
        summary["total_rows"] += sheet["shape"][0]
        summary["total_columns"] = max(summary["total_columns"], sheet["shape"][1])

def _analyze_data_quality(sheets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze data quality across all sheets"""
    quality_metrics = {
        "completeness": 0.0,
        "consistency": 0.0,
        "issues": []
    }
    
    total_cells = 0
    missing_cells = 0
    
    for sheet in sheets:
        rows, cols = sheet["shape"]
        sheet_cells = rows * cols
        total_cells += sheet_cells
        
        sheet_missing = sum(sheet["missing_values"].values())
        missing_cells += sheet_missing
        
        if sheet_missing / sheet_cells > 0.3:
            quality_metrics["issues"].append(f"Sheet '{sheet['name']}' has >30% missing values")
    
    if total_cells > 0:
        quality_metrics["completeness"] = 1.0 - (missing_cells / total_cells)
    
    quality_metrics["consistency"] = 1.0 if len(quality_metrics["issues"]) == 0 else 0.8
    
    return quality_metrics

async def _extract_spreadsheet_content(spreadsheet_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Legacy method for dictionary format"""
    try:
        content = spreadsheet_data.get("content", {})
        if isinstance(content, dict):
            if "url" in content:
                return await _download_and_process_spreadsheet(content["url"])
            elif "data" in content:
                filename = spreadsheet_data.get("filename", "spreadsheet.xlsx")
                return _process_spreadsheet_content(content["data"], filename)
    except Exception as e:
        logger.error(f"Error extracting spreadsheet content: {e}")
        return None
    
    return None

async def convert_spreadsheet_to_chart_data(sheet_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Convert spreadsheet data to chart-compatible format"""
    try:
        data = sheet_data.get("data", [])
        if not data:
            return None
        
        is_valid, message = DataValidator.validate_chart_data(data)
        if not is_valid:
            logger.warning(f"Spreadsheet data not suitable for charts: {message}")
            return None
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to convert spreadsheet to chart data: {e}")
        return None