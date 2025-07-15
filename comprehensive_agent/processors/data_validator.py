from typing import Dict, List, Any, Union, Tuple
import json
import re
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Exceptions & Data structures
# -----------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when data validation fails"""
    pass

@dataclass(slots=True)
class ValidationIssue:
    field: str
    issue: str  # e.g. 'null_value', 'invalid_numeric'
    severity: str  # 'warning' | 'error'

class DataValidator:
    @staticmethod
    def validate_financial_data(
        data: Dict[str, Any], *, required_fields: List[str] | None = None
    ) -> bool:
        """Validate basic structure of financial data.

        Parameters
        ----------
        data
            Input dict to validate.
        required_fields
            Override the default mandatory keys (symbol, price, volume).
        """

        if not isinstance(data, dict):
            raise ValidationError("Financial data must be a dictionary")
        
        if required_fields is None:
            required_fields = ["symbol", "price", "volume"]

        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")
        
        return DataValidator._validate_numeric_fields(data, ["price", "volume", "market_cap", "pe_ratio", "eps"])
    
    @staticmethod
    def sanitize_numeric_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and convert numeric fields"""
        sanitized = data.copy()
        
        for key, value in data.items():
            if DataValidator._is_numeric_field(key):
                sanitized[key] = DataValidator._clean_numeric_value(value)
        
        return sanitized
    
    @staticmethod
    def detect_data_quality_issues(data: Dict[str, Any]) -> List[ValidationIssue]:
        """Detect potential data quality issues"""
        issues = []
        
        for key, value in data.items():
            if value is None:
                issues.append(ValidationIssue(key, "null_value", "warning"))
            elif isinstance(value, str) and not value.strip():
                issues.append(ValidationIssue(key, "empty_string", "warning"))
            elif DataValidator._is_numeric_field(key) and not DataValidator._is_valid_number(value):
                issues.append(ValidationIssue(key, "invalid_numeric", "error"))
            elif DataValidator._is_date_field(key) and not DataValidator._is_valid_date(value):
                issues.append(ValidationIssue(key, "invalid_date", "error"))
        
        return issues
    
    @staticmethod
    def suggest_data_corrections(issues: List[ValidationIssue]) -> Dict[str, str]:
        """Suggest corrections for data quality issues"""
        corrections = {}
        
        for issue in issues:
            field = issue.field
            issue_type = issue.issue
            
            if issue_type == "null_value":
                corrections[field] = "Replace null with appropriate default value"
            elif issue_type == "empty_string":
                corrections[field] = "Provide non-empty string value"
            elif issue_type == "invalid_numeric":
                corrections[field] = "Ensure numeric value is valid (not NaN, Infinity)"
            elif issue_type == "invalid_date":
                corrections[field] = "Use ISO format (YYYY-MM-DD) for dates"
        
        return corrections
    
    @staticmethod
    def validate_widget_structure(widget_data: Any) -> Tuple[bool, str]:
        """Validate widget data structure"""
        if not widget_data:
            return False, "Widget data is empty"
        
        if isinstance(widget_data, list):
            for i, item in enumerate(widget_data):
                if not (hasattr(item, 'items') or isinstance(item, dict)):
                    return False, f"Invalid widget data structure at index {i}"
            return True, "Valid widget data list"
        
        if hasattr(widget_data, 'items'):
            if not hasattr(widget_data.items, '__iter__'):
                return False, "Widget items must be iterable"
            return True, "Valid DataContent structure"
        
        if isinstance(widget_data, dict):
            if "items" not in widget_data:
                return False, "Dictionary widget data missing 'items' field"
            return True, "Valid dictionary structure"
        
        return False, "Unrecognized widget data structure"
    
    @staticmethod
    def validate_chart_data(data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Validate data for chart generation"""
        if not data:
            return False, "Chart data is empty"
        
        if not isinstance(data, list):
            return False, "Chart data must be a list"
        
        if not all(isinstance(item, dict) for item in data):
            return False, "All chart data items must be dictionaries"
        
        if len(data) < 2:
            return False, "Insufficient data points for chart (minimum 2 required)"
        
        sample = data[0]
        if not sample:
            return False, "Empty data dictionary found"
        
        numeric_fields = [k for k, v in sample.items() if DataValidator._is_valid_number(v)]
        if not numeric_fields:
            return False, "No numeric fields found for chart visualization"
        
        return True, "Valid chart data"
    
    @staticmethod
    def _validate_numeric_fields(data: Dict[str, Any], numeric_fields: List[str]) -> bool:
        """Validate that numeric fields contain valid numbers"""
        for field in numeric_fields:
            if field in data and not DataValidator._is_valid_number(data[field]):
                raise ValidationError(f"Invalid numeric value for field '{field}': {data[field]}")
        
        return True
    
    @staticmethod
    def _is_numeric_field(field_name: str) -> bool:
        """Check if field should contain numeric data"""
        numeric_patterns = [
            r".*price.*", r".*volume.*", r".*ratio.*", r".*rate.*",
            r".*amount.*", r".*value.*", r".*cap.*", r".*eps.*"
        ]
        
        field_lower = field_name.lower()
        return any(re.match(pattern, field_lower) for pattern in numeric_patterns)
    
    @staticmethod
    def _is_date_field(field_name: str) -> bool:
        """Check if field should contain date data"""
        date_patterns = [r".*date.*", r".*time.*", r".*timestamp.*"]
        field_lower = field_name.lower()
        return any(re.match(pattern, field_lower) for pattern in date_patterns)
    
    @staticmethod
    def _is_valid_number(value: Any) -> bool:
        """Check if value is a valid number"""
        if value is None:
            return False
        
        if isinstance(value, (int, float)):
            return not (value != value or value == float('inf') or value == float('-inf'))
        
        if isinstance(value, str):
            cleaned = value.strip().lower()

            # Handle accounting negatives e.g. (1,234) -> -1234
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]

            # Remove common delimiter symbols
            cleaned = cleaned.replace(',', '').replace('$', '')

            # Handle scaled numbers like 1.2k, 3m, 4.5b, 2t
            scaled_value = DataValidator._parse_scaled_number(cleaned)
            if scaled_value is not None:
                return scaled_value

            # Fallback to plain float/int parsing
            try:
                return float(cleaned) if '.' in cleaned else int(cleaned)
            except (ValueError, TypeError):
                return False
        
        return False
    
    @staticmethod
    def _is_valid_date(value: Any) -> bool:
        """Check if value is a valid date"""
        if not isinstance(value, str):
            return False
        
        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            r"^\d{2}/\d{2}/\d{4}$"
        ]
        
        return any(re.match(pattern, value) for pattern in date_patterns)
    
    @staticmethod
    def _clean_numeric_value(value: Any) -> Union[float, int, None]:
        """Clean and convert numeric value"""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            if value != value or value == float('inf') or value == float('-inf'):
                return None
            return value
        
        if isinstance(value, str):
            cleaned = value.strip().lower()

            # Handle accounting negatives e.g. (1,234) -> -1234
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]

            # Remove common delimiter symbols
            cleaned = cleaned.replace(',', '').replace('$', '')

            # Handle scaled numbers like 1.2k, 3m, 4.5b, 2t
            scaled_value = DataValidator._parse_scaled_number(cleaned)
            if scaled_value is not None:
                return scaled_value

            # Fallback to plain float/int parsing
            try:
                return float(cleaned) if '.' in cleaned else int(cleaned)
            except (ValueError, TypeError):
                return None
        
        return None

    @staticmethod
    def _parse_scaled_number(value: str) -> float | int | None:
        """Convert suffixed numbers like '1.2k', '3m', '5b', '2t'."""

        multipliers = {
            'k': 1_000,
            'm': 1_000_000,
            'b': 1_000_000_000,
            't': 1_000_000_000_000,
        }

        if not value:
            return None

        suffix = value[-1]
        if suffix in multipliers:
            try:
                base_num = float(value[:-1])
                return base_num * multipliers[suffix]
            except ValueError:
                return None
         
        return None