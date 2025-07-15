from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from .error_handler import ErrorHandler, DataProcessingError, error_boundary
from .data_validator import DataValidator

logger = logging.getLogger(__name__)

class APIDataProcessor:
    """Enhanced API data processor with financial data normalization and time series handling."""
    
    def __init__(self):
        self.standard_financial_fields = {
            'price': ['price', 'close', 'closing_price', 'last_price', 'value'],
            'volume': ['volume', 'trading_volume', 'vol', 'quantity'],
            'date': ['date', 'timestamp', 'time', 'datetime', 'trading_date'],
            'open': ['open', 'opening_price', 'open_price'],
            'high': ['high', 'high_price', 'maximum'],
            'low': ['low', 'low_price', 'minimum'],
            'symbol': ['symbol', 'ticker', 'instrument', 'security'],
            'market_cap': ['market_cap', 'marketcap', 'market_capitalization'],
            'pe_ratio': ['pe_ratio', 'p_e_ratio', 'price_earnings_ratio'],
            'eps': ['eps', 'earnings_per_share', 'earning_per_share']
        }
        
        self.time_series_patterns = [
            'daily', 'weekly', 'monthly', 'quarterly', 'yearly',
            'intraday', 'minute', 'hourly'
        ]
    
    @error_boundary(DataProcessingError)
    async def normalize_financial_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize data formats across different sources."""
        try:
            normalized = {
                'data': [],
                'metadata': {
                    'source_format': self._detect_source_format(raw_data),
                    'normalization_applied': [],
                    'processed_at': datetime.now().isoformat()
                },
                'schema': {
                    'standardized_fields': {},
                    'original_fields': [],
                    'field_mappings': {}
                }
            }
            
            # Extract raw records
            records = self._extract_records(raw_data)
            if not records:
                return normalized
            
            # Normalize field names and values
            normalized_records = []
            field_mappings = {}
            
            for record in records:
                normalized_record, mappings = self._normalize_record(record)
                normalized_records.append(normalized_record)
                field_mappings.update(mappings)
            
            # Apply data type standardization
            standardized_records = self._standardize_data_types(normalized_records)
            
            normalized.update({
                'data': standardized_records,
                'metadata': {
                    **normalized['metadata'],
                    'total_records': len(standardized_records),
                    'normalization_applied': ['field_mapping', 'type_conversion', 'value_cleaning']
                },
                'schema': {
                    'standardized_fields': self._get_schema_info(standardized_records),
                    'original_fields': list(set().union(*[record.keys() for record in records])),
                    'field_mappings': field_mappings
                }
            })
            
            return normalized
            
        except Exception as e:
            await ErrorHandler.log_error_with_context(
                DataProcessingError(f"Data normalization failed: {e}"), 
                raw_data
            )
            return {}
    
    @error_boundary(DataProcessingError)
    async def merge_time_series(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Intelligent merging of time series data."""
        try:
            if not datasets or len(datasets) < 2:
                return datasets[0] if datasets else {}
            
            merged = {
                'data': [],
                'metadata': {
                    'source_datasets': len(datasets),
                    'merge_strategy': 'intelligent_time_series',
                    'processed_at': datetime.now().isoformat()
                },
                'merge_info': {
                    'time_ranges': [],
                    'overlapping_periods': [],
                    'data_quality': {}
                }
            }
            
            # Convert datasets to DataFrames for easier manipulation
            dataframes = []
            time_columns = []
            
            for i, dataset in enumerate(datasets):
                df, time_col = self._dataset_to_dataframe(dataset, f"dataset_{i}")
                if df is not None:
                    dataframes.append(df)
                    time_columns.append(time_col)
                    
                    # Record time range
                    if time_col and time_col in df.columns:
                        time_range = {
                            'dataset_index': i,
                            'start_date': df[time_col].min(),
                            'end_date': df[time_col].max(),
                            'frequency': self._detect_frequency(df[time_col])
                        }
                        merged['merge_info']['time_ranges'].append(time_range)
            
            if not dataframes:
                return merged
            
            # Merge DataFrames intelligently
            merged_df = self._intelligent_merge(dataframes, time_columns)
            
            # Detect and handle overlapping periods
            overlaps = self._detect_overlapping_periods(merged['merge_info']['time_ranges'])
            merged['merge_info']['overlapping_periods'] = overlaps
            
            # Convert back to records
            merged['data'] = merged_df.to_dict('records')
            merged['metadata']['total_records'] = len(merged['data'])
            
            return merged
            
        except Exception as e:
            await ErrorHandler.log_error_with_context(
                DataProcessingError(f"Time series merging failed: {e}"), 
                {'datasets_count': len(datasets)}
            )
            return {}
    
    @error_boundary(DataProcessingError)
    async def fill_missing_data(self, data: Dict[str, Any], method: str = "interpolate") -> Dict[str, Any]:
        """Handle missing data points intelligently."""
        try:
            filled_data = {
                'data': [],
                'metadata': {
                    'fill_method': method,
                    'missing_data_stats': {},
                    'processed_at': datetime.now().isoformat()
                },
                'fill_info': {
                    'fields_processed': [],
                    'values_filled': 0,
                    'quality_improvement': {}
                }
            }
            
            records = data.get('data', [])
            if not records:
                return filled_data
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(records)
            original_missing = df.isnull().sum()
            
            # Apply filling strategy based on data types
            filled_df = self._apply_filling_strategy(df, method)
            
            # Calculate improvement metrics
            final_missing = filled_df.isnull().sum()
            values_filled = (original_missing - final_missing).sum()
            
            filled_data.update({
                'data': filled_df.to_dict('records'),
                'metadata': {
                    **filled_data['metadata'],
                    'total_records': len(filled_df),
                    'missing_data_stats': {
                        'original_missing': original_missing.to_dict(),
                        'final_missing': final_missing.to_dict(),
                        'improvement_ratio': ((original_missing - final_missing) / original_missing.replace(0, 1)).to_dict()
                    }
                },
                'fill_info': {
                    'fields_processed': list(df.columns),
                    'values_filled': int(values_filled),
                    'quality_improvement': {
                        'completeness_before': 1 - (original_missing.sum() / df.size),
                        'completeness_after': 1 - (final_missing.sum() / filled_df.size)
                    }
                }
            })
            
            return filled_data
            
        except Exception as e:
            await ErrorHandler.log_error_with_context(
                DataProcessingError(f"Missing data filling failed: {e}"), 
                {'method': method}
            )
            return data
    
    def _detect_source_format(self, raw_data: Dict[str, Any]) -> str:
        """Detect the format/source of raw data."""
        if 'items' in raw_data:
            return 'openbb_widget'
        elif 'data' in raw_data:
            return 'structured_api'
        elif isinstance(raw_data, list):
            return 'record_list'
        elif any(key in raw_data for key in ['price', 'volume', 'symbol']):
            return 'financial_record'
        else:
            return 'generic_dict'
    
    def _extract_records(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract records from various data formats."""
        if isinstance(raw_data, list):
            return raw_data
        
        if 'data' in raw_data:
            data = raw_data['data']
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        
        if 'items' in raw_data:
            return raw_data['items']
        
        # Treat single dict as single record
        if isinstance(raw_data, dict):
            return [raw_data]
        
        return []
    
    def _normalize_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Normalize a single record and return field mappings."""
        normalized = {}
        mappings = {}
        
        for original_field, value in record.items():
            # Find standard field mapping
            standard_field = self._map_to_standard_field(original_field)
            
            if standard_field:
                # Clean and convert value
                cleaned_value = self._clean_field_value(value, standard_field)
                normalized[standard_field] = cleaned_value
                mappings[original_field] = standard_field
            else:
                # Keep original field but clean value
                cleaned_field = self._clean_field_name(original_field)
                normalized[cleaned_field] = self._clean_field_value(value, 'generic')
                mappings[original_field] = cleaned_field
        
        return normalized, mappings
    
    def _map_to_standard_field(self, field_name: str) -> Optional[str]:
        """Map field name to standard financial field."""
        field_lower = field_name.lower().replace('_', '').replace(' ', '')
        
        for standard_field, variants in self.standard_financial_fields.items():
            for variant in variants:
                variant_clean = variant.lower().replace('_', '').replace(' ', '')
                if variant_clean in field_lower or field_lower in variant_clean:
                    return standard_field
        
        return None
    
    def _clean_field_name(self, field_name: str) -> str:
        """Clean and standardize field name."""
        # Convert to lowercase, replace spaces/hyphens with underscores
        cleaned = field_name.lower().replace(' ', '_').replace('-', '_')
        # Remove special characters except underscores
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
        # Remove leading/trailing underscores
        return cleaned.strip('_')
    
    def _clean_field_value(self, value: Any, field_type: str) -> Any:
        """Clean and convert field value based on its type."""
        if value is None:
            return None
        
        if field_type in ['price', 'volume', 'market_cap', 'pe_ratio', 'eps', 'open', 'high', 'low']:
            return DataValidator._clean_numeric_value(value)
        elif field_type == 'date':
            return self._clean_date_value(value)
        elif field_type == 'symbol':
            return str(value).upper().strip() if value else None
        else:
            # Generic cleaning
            if isinstance(value, str):
                return value.strip()
            return value
    
    def _clean_date_value(self, value: Any) -> Optional[str]:
        """Clean and standardize date value."""
        if value is None:
            return None
        
        if isinstance(value, datetime):
            return value.isoformat()
        
        if isinstance(value, str):
            # Try to parse common date formats
            date_formats = [
                '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d'
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(value.strip(), fmt)
                    return parsed_date.isoformat()
                except ValueError:
                    continue
        
        return str(value) if value else None
    
    def _standardize_data_types(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize data types across all records."""
        if not records:
            return records
        
        # Infer consistent types for each field
        type_inference = {}
        
        for field in records[0].keys():
            values = [record.get(field) for record in records if record.get(field) is not None]
            if values:
                type_inference[field] = self._infer_consistent_type(values)
        
        # Apply type standardization
        standardized = []
        for record in records:
            standardized_record = {}
            for field, value in record.items():
                if field in type_inference:
                    standardized_record[field] = self._convert_to_type(value, type_inference[field])
                else:
                    standardized_record[field] = value
            standardized.append(standardized_record)
        
        return standardized
    
    def _infer_consistent_type(self, values: List[Any]) -> str:
        """Infer the most consistent type for a list of values."""
        type_counts = {}
        
        for value in values:
            if DataValidator._is_valid_number(value):
                type_counts['numeric'] = type_counts.get('numeric', 0) + 1
            elif self._is_date_like(value):
                type_counts['date'] = type_counts.get('date', 0) + 1
            else:
                type_counts['string'] = type_counts.get('string', 0) + 1
        
        return max(type_counts, key=type_counts.get) if type_counts else 'string'
    
    def _convert_to_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target type."""
        if value is None:
            return None
        
        try:
            if target_type == 'numeric':
                return DataValidator._clean_numeric_value(value)
            elif target_type == 'date':
                return self._clean_date_value(value)
            else:
                return str(value)
        except:
            return value
    
    def _get_schema_info(self, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Get schema information for standardized records."""
        schema = {}
        
        if not records:
            return schema
        
        for field in records[0].keys():
            values = [record.get(field) for record in records if record.get(field) is not None]
            
            schema[field] = {
                'type': self._infer_consistent_type(values),
                'nullable': len(values) < len(records),
                'unique_count': len(set(str(v) for v in values)),
                'sample_values': list(set(str(v) for v in values))[:3]
            }
        
        return schema
    
    def _dataset_to_dataframe(self, dataset: Dict[str, Any], dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Convert dataset to DataFrame and identify time column."""
        records = dataset.get('data', [])
        if not records:
            return None, None
        
        df = pd.DataFrame(records)
        
        # Identify time column
        time_column = self._identify_time_column(df)
        
        if time_column:
            # Convert time column to datetime
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            # Sort by time
            df = df.sort_values(time_column)
        
        # Add dataset identifier
        df['_source_dataset'] = dataset_name
        
        return df, time_column
    
    def _identify_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the time/date column in DataFrame."""
        # Check for standard time field names
        time_candidates = ['date', 'time', 'timestamp', 'datetime']
        
        for col in df.columns:
            if any(candidate in col.lower() for candidate in time_candidates):
                return col
        
        # Check for datetime-like content
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(10)
                if len(sample) > 0 and self._is_date_like(sample.iloc[0]):
                    return col
        
        return None
    
    def _is_date_like(self, value: Any) -> bool:
        """Check if value looks like a date."""
        if isinstance(value, datetime):
            return True
        
        if isinstance(value, str):
            return DataValidator._is_valid_date(value)
        
        return False
    
    def _detect_frequency(self, time_series: pd.Series) -> str:
        """Detect the frequency of a time series."""
        if len(time_series) < 2:
            return 'unknown'
        
        # Calculate time differences
        time_diffs = time_series.sort_values().diff().dropna()
        
        if len(time_diffs) == 0:
            return 'unknown'
        
        # Get median time difference
        median_diff = time_diffs.median()
        
        if median_diff <= timedelta(minutes=5):
            return 'minute'
        elif median_diff <= timedelta(hours=2):
            return 'hourly'
        elif median_diff <= timedelta(days=1.5):
            return 'daily'
        elif median_diff <= timedelta(days=8):
            return 'weekly'
        elif median_diff <= timedelta(days=35):
            return 'monthly'
        elif median_diff <= timedelta(days=100):
            return 'quarterly'
        else:
            return 'yearly'
    
    def _intelligent_merge(self, dataframes: List[pd.DataFrame], time_columns: List[str]) -> pd.DataFrame:
        """Intelligently merge multiple DataFrames."""
        if not dataframes:
            return pd.DataFrame()
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Start with the first DataFrame
        merged = dataframes[0].copy()
        
        for i, df in enumerate(dataframes[1:], 1):
            time_col = time_columns[i] if i < len(time_columns) else None
            base_time_col = time_columns[0] if time_columns else None
            
            if time_col and base_time_col and time_col in df.columns and base_time_col in merged.columns:
                # Time-based merge
                merged = self._merge_on_time(merged, df, base_time_col, time_col)
            else:
                # Append merge (concatenation)
                merged = pd.concat([merged, df], ignore_index=True, sort=False)
        
        return merged
    
    def _merge_on_time(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                      left_time_col: str, right_time_col: str) -> pd.DataFrame:
        """Merge DataFrames on time columns with intelligent overlap handling."""
        # Rename time columns to match
        right_df_copy = right_df.copy()
        if right_time_col != left_time_col:
            right_df_copy = right_df_copy.rename(columns={right_time_col: left_time_col})
        
        # Merge with outer join to keep all time points
        merged = pd.merge(left_df, right_df_copy, on=left_time_col, how='outer', suffixes=('', '_right'))
        
        # Handle overlapping columns intelligently
        for col in merged.columns:
            if col.endswith('_right'):
                base_col = col[:-6]  # Remove '_right' suffix
                if base_col in merged.columns:
                    # Fill missing values from the right DataFrame
                    merged[base_col] = merged[base_col].combine_first(merged[col])
                    merged = merged.drop(columns=[col])
                else:
                    # Rename right column to base name
                    merged = merged.rename(columns={col: base_col})
        
        return merged.sort_values(left_time_col)
    
    def _detect_overlapping_periods(self, time_ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect overlapping time periods between datasets."""
        overlaps = []
        
        for i, range1 in enumerate(time_ranges):
            for j, range2 in enumerate(time_ranges[i+1:], i+1):
                start1, end1 = range1['start_date'], range1['end_date']
                start2, end2 = range2['start_date'], range2['end_date']
                
                # Check for overlap
                if start1 <= end2 and start2 <= end1:
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    
                    overlaps.append({
                        'dataset_1': range1['dataset_index'],
                        'dataset_2': range2['dataset_index'],
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'overlap_duration': overlap_end - overlap_start
                    })
        
        return overlaps
    
    def _apply_filling_strategy(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply missing data filling strategy."""
        filled_df = df.copy()
        
        for col in filled_df.columns:
            if filled_df[col].isnull().any():
                if method == "interpolate":
                    filled_df[col] = self._interpolate_column(filled_df[col])
                elif method == "forward_fill":
                    filled_df[col] = filled_df[col].fillna(method='ffill')
                elif method == "backward_fill":
                    filled_df[col] = filled_df[col].fillna(method='bfill')
                elif method == "mean":
                    if pd.api.types.is_numeric_dtype(filled_df[col]):
                        filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
                elif method == "median":
                    if pd.api.types.is_numeric_dtype(filled_df[col]):
                        filled_df[col] = filled_df[col].fillna(filled_df[col].median())
                elif method == "mode":
                    mode_val = filled_df[col].mode()
                    if len(mode_val) > 0:
                        filled_df[col] = filled_df[col].fillna(mode_val[0])
        
        return filled_df
    
    def _interpolate_column(self, series: pd.Series) -> pd.Series:
        """Intelligently interpolate column values."""
        if pd.api.types.is_numeric_dtype(series):
            # Linear interpolation for numeric data
            return series.interpolate(method='linear')
        else:
            # Forward fill for non-numeric data
            return series.fillna(method='ffill').fillna(method='bfill')