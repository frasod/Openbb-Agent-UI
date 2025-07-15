from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
import logging
from .error_handler import ErrorHandler, DataProcessingError, error_boundary
from .data_validator import DataValidator

logger = logging.getLogger(__name__)

class SpreadsheetProcessor:
    """Enhanced spreadsheet processor with financial statement detection and metrics calculation."""
    
    def __init__(self):
        self.financial_statement_keywords = {
            'income_statement': [
                'revenue', 'sales', 'income', 'profit', 'loss', 'ebitda', 'ebit',
                'cost of goods sold', 'cogs', 'operating expenses', 'net income',
                'gross profit', 'operating income', 'earnings'
            ],
            'balance_sheet': [
                'assets', 'liabilities', 'equity', 'cash', 'inventory', 'receivables',
                'current assets', 'fixed assets', 'current liabilities', 'long term debt',
                'shareholders equity', 'retained earnings', 'total assets'
            ],
            'cash_flow': [
                'cash flow', 'operating activities', 'investing activities',
                'financing activities', 'free cash flow', 'capex', 'depreciation',
                'working capital', 'cash from operations'
            ]
        }
    
    @error_boundary(DataProcessingError)
    async def parse_excel_file(self, file_content: bytes) -> Dict[str, Any]:
        """Parse Excel with multiple sheets, detect financial tables."""
        try:
            data_io = io.BytesIO(file_content)
            excel_file = pd.ExcelFile(data_io)
            
            parsed_data = {
                'sheets': {},
                'financial_statements': {},
                'metadata': {
                    'total_sheets': len(excel_file.sheet_names),
                    'sheet_names': excel_file.sheet_names,
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            for sheet_name in excel_file.sheet_names:
                sheet_data = await self._process_excel_sheet(excel_file, sheet_name)
                if sheet_data:
                    parsed_data['sheets'][sheet_name] = sheet_data
            
            if parsed_data['sheets']:
                parsed_data['financial_statements'] = await self.detect_financial_statements(parsed_data['sheets'])
                
            return parsed_data
            
        except Exception as e:
            await ErrorHandler.log_error_with_context(
                DataProcessingError(f"Excel parsing failed: {e}"), 
                {'file_size': len(file_content)}
            )
            return {}
    
    async def _process_excel_sheet(self, excel_file: pd.ExcelFile, sheet_name: str) -> Optional[Dict[str, Any]]:
        """Process individual Excel sheet with financial data detection."""
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            if df.empty:
                return None
            
            # Clean and prepare data
            df_clean = self._clean_dataframe(df)
            
            # Detect financial data patterns
            financial_patterns = self._detect_financial_patterns(df_clean, sheet_name)
            
            sheet_data = {
                'name': sheet_name,
                'data': df_clean.to_dict('records'),
                'shape': df_clean.shape,
                'columns': df_clean.columns.tolist(),
                'dtypes': df_clean.dtypes.astype(str).to_dict(),
                'summary_stats': self._generate_enhanced_stats(df_clean),
                'financial_patterns': financial_patterns,
                'data_quality': self._assess_data_quality(df_clean)
            }
            
            return sheet_data
            
        except Exception as e:
            logger.error(f"Failed to process sheet '{sheet_name}': {e}")
            return None
    
    async def detect_financial_statements(self, sheets: Dict[str, Any]) -> Dict[str, Any]:
        """Identify P&L, Balance Sheet, Cash Flow statements."""
        statements = {
            'income_statement': None,
            'balance_sheet': None,
            'cash_flow_statement': None,
            'other_financial_data': []
        }
        
        for sheet_name, sheet_data in sheets.items():
            statement_type = self._classify_financial_statement(sheet_data)
            
            if statement_type in statements and statements[statement_type] is None:
                statements[statement_type] = {
                    'sheet_name': sheet_name,
                    'data': sheet_data['data'],
                    'confidence_score': sheet_data['financial_patterns']['confidence_score'],
                    'detected_metrics': sheet_data['financial_patterns']['detected_metrics']
                }
            elif statement_type == 'financial_data':
                statements['other_financial_data'].append({
                    'sheet_name': sheet_name,
                    'data': sheet_data['data'],
                    'patterns': sheet_data['financial_patterns']
                })
        
        return statements
    
    async def extract_financial_metrics(self, statements: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key ratios and metrics automatically."""
        metrics = {
            'profitability_ratios': {},
            'liquidity_ratios': {},
            'leverage_ratios': {},
            'efficiency_ratios': {},
            'valuation_metrics': {},
            'calculated_at': datetime.now().isoformat()
        }
        
        try:
            # Extract metrics from income statement
            if statements.get('income_statement'):
                income_data = statements['income_statement']['data']
                metrics['profitability_ratios'].update(
                    self._calculate_profitability_ratios(income_data)
                )
            
            # Extract metrics from balance sheet
            if statements.get('balance_sheet'):
                balance_data = statements['balance_sheet']['data']
                metrics['liquidity_ratios'].update(
                    self._calculate_liquidity_ratios(balance_data)
                )
                metrics['leverage_ratios'].update(
                    self._calculate_leverage_ratios(balance_data)
                )
            
            # Cross-statement ratios
            if statements.get('income_statement') and statements.get('balance_sheet'):
                income_data = statements['income_statement']['data']
                balance_data = statements['balance_sheet']['data']
                metrics['efficiency_ratios'].update(
                    self._calculate_efficiency_ratios(income_data, balance_data)
                )
            
        except Exception as e:
            logger.error(f"Failed to extract financial metrics: {e}")
        
        return metrics
    
    @error_boundary(DataProcessingError)
    async def handle_csv_data(self, csv_content: str) -> Dict[str, Any]:
        """Smart CSV parsing with type detection."""
        try:
            # Detect delimiter and encoding
            delimiter = self._detect_csv_delimiter(csv_content)
            
            # Parse CSV with detected settings
            df = pd.read_csv(io.StringIO(csv_content), delimiter=delimiter)
            
            if df.empty:
                return {}
            
            # Clean and process data
            df_clean = self._clean_dataframe(df)
            
            # Detect financial patterns
            financial_patterns = self._detect_financial_patterns(df_clean, "csv_data")
            
            parsed_data = {
                'data': df_clean.to_dict('records'),
                'shape': df_clean.shape,
                'columns': df_clean.columns.tolist(),
                'dtypes': df_clean.dtypes.astype(str).to_dict(),
                'delimiter': delimiter,
                'summary_stats': self._generate_enhanced_stats(df_clean),
                'financial_patterns': financial_patterns,
                'data_quality': self._assess_data_quality(df_clean),
                'metadata': {
                    'source_type': 'csv',
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            return parsed_data
            
        except Exception as e:
            await ErrorHandler.log_error_with_context(
                DataProcessingError(f"CSV parsing failed: {e}"), 
                {'content_length': len(csv_content)}
            )
            return {}
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame."""
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df_clean.columns = [
            str(col).strip().lower().replace(' ', '_').replace('-', '_')
            for col in df_clean.columns
        ]
        
        # Convert numeric columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                numeric_col = pd.to_numeric(df_clean[col], errors='ignore')
                if not numeric_col.equals(df_clean[col]):
                    df_clean[col] = numeric_col
        
        return df_clean
    
    def _detect_financial_patterns(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Detect financial data patterns in DataFrame."""
        patterns = {
            'is_financial_data': False,
            'statement_type': 'unknown',
            'confidence_score': 0.0,
            'detected_metrics': [],
            'time_series_columns': [],
            'value_columns': []
        }
        
        # Check column names for financial keywords
        columns_text = ' '.join(df.columns).lower()
        sheet_text = sheet_name.lower()
        
        financial_score = 0
        detected_metrics = []
        
        # Check for financial keywords
        for category, keywords in self.financial_statement_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in columns_text or keyword in sheet_text)
            if matches > 0:
                financial_score += matches
                detected_metrics.extend([kw for kw in keywords if kw in columns_text or kw in sheet_text])
        
        # Detect time series patterns
        date_columns = [col for col in df.columns if self._is_date_column(df[col])]
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        patterns.update({
            'is_financial_data': financial_score > 2,
            'confidence_score': min(financial_score / 10, 1.0),
            'detected_metrics': list(set(detected_metrics)),
            'time_series_columns': date_columns,
            'value_columns': numeric_columns
        })
        
        return patterns
    
    def _classify_financial_statement(self, sheet_data: Dict[str, Any]) -> str:
        """Classify the type of financial statement."""
        patterns = sheet_data.get('financial_patterns', {})
        detected_metrics = patterns.get('detected_metrics', [])
        
        if not patterns.get('is_financial_data'):
            return 'non_financial'
        
        # Score each statement type
        scores = {}
        for statement_type, keywords in self.financial_statement_keywords.items():
            score = sum(1 for metric in detected_metrics if any(kw in metric for kw in keywords))
            scores[statement_type] = score
        
        if not scores or max(scores.values()) == 0:
            return 'financial_data'
        
        return max(scores, key=scores.get)
    
    def _calculate_profitability_ratios(self, income_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate profitability ratios from income statement data."""
        ratios = {}
        
        try:
            # Extract key values
            revenue = self._extract_financial_value(income_data, ['revenue', 'sales', 'total_revenue'])
            gross_profit = self._extract_financial_value(income_data, ['gross_profit', 'gross_income'])
            net_income = self._extract_financial_value(income_data, ['net_income', 'net_profit', 'profit'])
            ebitda = self._extract_financial_value(income_data, ['ebitda'])
            
            if revenue and revenue != 0:
                if gross_profit is not None:
                    ratios['gross_margin'] = gross_profit / revenue
                if net_income is not None:
                    ratios['net_margin'] = net_income / revenue
                if ebitda is not None:
                    ratios['ebitda_margin'] = ebitda / revenue
            
        except Exception as e:
            logger.error(f"Error calculating profitability ratios: {e}")
        
        return ratios
    
    def _calculate_liquidity_ratios(self, balance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate liquidity ratios from balance sheet data."""
        ratios = {}
        
        try:
            current_assets = self._extract_financial_value(balance_data, ['current_assets', 'total_current_assets'])
            current_liabilities = self._extract_financial_value(balance_data, ['current_liabilities', 'total_current_liabilities'])
            cash = self._extract_financial_value(balance_data, ['cash', 'cash_and_equivalents'])
            
            if current_assets and current_liabilities and current_liabilities != 0:
                ratios['current_ratio'] = current_assets / current_liabilities
                
                if cash is not None:
                    ratios['cash_ratio'] = cash / current_liabilities
            
        except Exception as e:
            logger.error(f"Error calculating liquidity ratios: {e}")
        
        return ratios
    
    def _calculate_leverage_ratios(self, balance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate leverage ratios from balance sheet data."""
        ratios = {}
        
        try:
            total_debt = self._extract_financial_value(balance_data, ['total_debt', 'total_liabilities'])
            total_equity = self._extract_financial_value(balance_data, ['total_equity', 'shareholders_equity'])
            total_assets = self._extract_financial_value(balance_data, ['total_assets'])
            
            if total_debt and total_equity and total_equity != 0:
                ratios['debt_to_equity'] = total_debt / total_equity
            
            if total_debt and total_assets and total_assets != 0:
                ratios['debt_to_assets'] = total_debt / total_assets
            
        except Exception as e:
            logger.error(f"Error calculating leverage ratios: {e}")
        
        return ratios
    
    def _calculate_efficiency_ratios(self, income_data: List[Dict[str, Any]], balance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate efficiency ratios using both income and balance sheet data."""
        ratios = {}
        
        try:
            revenue = self._extract_financial_value(income_data, ['revenue', 'sales'])
            total_assets = self._extract_financial_value(balance_data, ['total_assets'])
            net_income = self._extract_financial_value(income_data, ['net_income', 'net_profit'])
            
            if revenue and total_assets and total_assets != 0:
                ratios['asset_turnover'] = revenue / total_assets
            
            if net_income and total_assets and total_assets != 0:
                ratios['roa'] = net_income / total_assets
            
        except Exception as e:
            logger.error(f"Error calculating efficiency ratios: {e}")
        
        return ratios
    
    def _extract_financial_value(self, data: List[Dict[str, Any]], possible_keys: List[str]) -> Optional[float]:
        """Extract financial value from data using possible key names."""
        for row in data:
            for key in possible_keys:
                for actual_key, value in row.items():
                    if key.lower() in actual_key.lower():
                        cleaned_value = DataValidator._clean_numeric_value(value)
                        if cleaned_value is not None:
                            return float(cleaned_value)
        return None
    
    def _generate_enhanced_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate enhanced summary statistics."""
        stats = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric'] = df[numeric_cols].describe().to_dict()
            stats['correlations'] = df[numeric_cols].corr().to_dict()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats['categorical'] = {}
            for col in categorical_cols:
                stats['categorical'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(3).to_dict()
                }
        
        return stats
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        return {
            'completeness': 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0,
            'missing_values_by_column': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types_consistency': self._check_type_consistency(df)
        }
    
    def _check_type_consistency(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check if data types are consistent within columns."""
        consistency = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric-looking strings are consistent
                numeric_count = sum(1 for val in df[col].dropna() if DataValidator._is_valid_number(val))
                total_count = len(df[col].dropna())
                consistency[col] = numeric_count == 0 or numeric_count == total_count
            else:
                consistency[col] = True
        
        return consistency
    
    def _detect_csv_delimiter(self, csv_content: str) -> str:
        """Detect CSV delimiter."""
        sample = csv_content[:1000]
        delimiters = [',', ';', '\t', '|']
        
        delimiter_counts = {delim: sample.count(delim) for delim in delimiters}
        return max(delimiter_counts, key=delimiter_counts.get) if max(delimiter_counts.values()) > 0 else ','
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a series contains date data."""
        if series.dtype == 'datetime64[ns]':
            return True
        
        if series.dtype == 'object':
            sample = series.dropna().head(10)
            date_count = sum(1 for val in sample if DataValidator._is_valid_date(str(val)))
            return date_count > len(sample) * 0.8
        
        return False