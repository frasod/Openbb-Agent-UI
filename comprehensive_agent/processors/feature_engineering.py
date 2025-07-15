from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .error_handler import error_boundary, DataProcessingError

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    enable_technical_indicators: bool = True
    enable_statistical_features: bool = True
    enable_time_features: bool = True
    enable_fundamental_ratios: bool = True
    lookback_periods: List[int] = None
    volatility_windows: List[int] = None

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 30]


class FeatureEngineer:
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_registry = {}

    @error_boundary(fallback={})
    async def engineer_features(self, data: Union[Dict, pd.DataFrame, List[Dict]]) -> Dict[str, Any]:
        df = await self._prepare_dataframe(data)
        
        if df.empty:
            return {"status": "error", "message": "No data to process"}

        features = {"status": "success", "features": {}}
        
        if self.config.enable_technical_indicators:
            features["features"].update(await self._generate_technical_indicators(df))
        
        if self.config.enable_statistical_features:
            features["features"].update(await self._generate_statistical_features(df))
        
        if self.config.enable_time_features:
            features["features"].update(await self._generate_time_features(df))
        
        if self.config.enable_fundamental_ratios:
            features["features"].update(await self._generate_fundamental_features(df))

        features["feature_count"] = len(features["features"])
        features["data_points"] = len(df)
        features["generated_at"] = datetime.now().isoformat()
        
        return features

    async def _prepare_dataframe(self, data: Union[Dict, pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        if isinstance(data, dict):
            try:
                df = pd.DataFrame([data])
            except ValueError:
                df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise DataProcessingError(f"Unsupported data type: {type(data)}")

        await self._standardize_column_names(df)
        await self._handle_missing_values(df)
        await self._validate_required_columns(df)
        
        return df

    async def _standardize_column_names(self, df: pd.DataFrame) -> None:
        column_mapping = {
            'open': ['Open', 'OPEN', 'open_price'],
            'high': ['High', 'HIGH', 'high_price'],
            'low': ['Low', 'LOW', 'low_price'],
            'close': ['Close', 'CLOSE', 'close_price', 'price'],
            'volume': ['Volume', 'VOLUME', 'vol'],
            'date': ['Date', 'DATE', 'timestamp', 'time'],
            'symbol': ['Symbol', 'SYMBOL', 'ticker', 'stock']
        }
        
        for standard_name, variations in column_mapping.items():
            for col in df.columns:
                if col in variations:
                    df.rename(columns={col: standard_name}, inplace=True)
                    break

    async def _handle_missing_values(self, df: pd.DataFrame) -> None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().any():
                if col in ['open', 'high', 'low', 'close']:
                    df[col].fillna(method='ffill', inplace=True)
                elif col == 'volume':
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)

    async def _validate_required_columns(self, df: pd.DataFrame) -> None:
        if 'close' not in df.columns and 'price' not in df.columns:
            raise DataProcessingError("No price data found (close or price column required)")

    @error_boundary(fallback={})
    async def _generate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        if 'close' in df.columns:
            price_col = 'close'
        elif 'price' in df.columns:
            price_col = 'price'
        else:
            return features

        prices = df[price_col].values
        
        for period in self.config.lookback_periods:
            if len(prices) >= period:
                features[f'sma_{period}'] = await self._simple_moving_average(prices, period)
                features[f'ema_{period}'] = await self._exponential_moving_average(prices, period)
                features[f'price_change_{period}'] = await self._price_change(prices, period)
                features[f'volatility_{period}'] = await self._volatility(prices, period)

        if len(prices) >= 14:
            features['rsi_14'] = await self._relative_strength_index(prices, 14)
        
        if 'volume' in df.columns and len(df) >= 20:
            volume = df['volume'].values
            features['volume_sma_20'] = await self._simple_moving_average(volume, 20)
            features['volume_ratio'] = volume[-1] / features['volume_sma_20'] if features['volume_sma_20'] > 0 else 0

        if all(col in df.columns for col in ['high', 'low', 'close']) and len(df) >= 20:
            features.update(await self._bollinger_bands(df, 20))

        return features

    @error_boundary(fallback={})
    async def _generate_statistical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if len(df[col].dropna()) > 0:
                values = df[col].dropna().values
                features[f'{col}_mean'] = np.mean(values)
                features[f'{col}_std'] = np.std(values)
                features[f'{col}_skew'] = await self._skewness(values)
                features[f'{col}_kurtosis'] = await self._kurtosis(values)
                
                if len(values) > 1:
                    features[f'{col}_min'] = np.min(values)
                    features[f'{col}_max'] = np.max(values)
                    features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']

        return features

    @error_boundary(fallback={})
    async def _generate_time_features(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                latest_date = df['date'].max()
                
                features['day_of_week'] = latest_date.dayofweek
                features['day_of_month'] = latest_date.day
                features['month'] = latest_date.month
                features['quarter'] = latest_date.quarter
                features['is_month_end'] = float(latest_date == latest_date + pd.offsets.MonthEnd(0))
                features['is_quarter_end'] = float(latest_date.month % 3 == 0 and latest_date == latest_date + pd.offsets.MonthEnd(0))
                
            except Exception as e:
                logger.warning(f"Could not process date features: {e}")

        return features

    @error_boundary(fallback={})
    async def _generate_fundamental_features(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        fundamental_ratios = {
            'pe_ratio': lambda row: row.get('price', row.get('close', 0)) / row.get('eps', 1) if row.get('eps', 0) != 0 else 0,
            'pb_ratio': lambda row: row.get('price', row.get('close', 0)) / row.get('book_value_per_share', 1) if row.get('book_value_per_share', 0) != 0 else 0,
            'debt_to_equity': lambda row: row.get('total_debt', 0) / row.get('total_equity', 1) if row.get('total_equity', 0) != 0 else 0,
            'current_ratio': lambda row: row.get('current_assets', 0) / row.get('current_liabilities', 1) if row.get('current_liabilities', 0) != 0 else 0,
            'roe': lambda row: row.get('net_income', 0) / row.get('shareholder_equity', 1) if row.get('shareholder_equity', 0) != 0 else 0
        }
        
        if not df.empty:
            latest_row = df.iloc[-1].to_dict()
            
            for ratio_name, calculation in fundamental_ratios.items():
                try:
                    features[ratio_name] = calculation(latest_row)
                except (ZeroDivisionError, TypeError, KeyError):
                    features[ratio_name] = 0.0

        return features

    async def _simple_moving_average(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return 0.0
        return np.mean(prices[-period:])

    async def _exponential_moving_average(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return 0.0
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema

    async def _price_change(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        return (prices[-1] - prices[-period-1]) / prices[-period-1] if prices[-period-1] != 0 else 0.0

    async def _volatility(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        
        returns = np.diff(prices[-period-1:]) / prices[-period-1:-1]
        return np.std(returns) * np.sqrt(252)

    async def _relative_strength_index(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    async def _bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        if len(df) < period:
            return {}
        
        close_prices = df['close'].values[-period:]
        sma = np.mean(close_prices)
        std = np.std(close_prices)
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        current_price = close_prices[-1]
        
        return {
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_middle': sma,
            'bb_width': upper_band - lower_band,
            'bb_position': (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        }

    async def _skewness(self, values: np.ndarray) -> float:
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        normalized = (values - mean) / std
        return np.mean(normalized ** 3)

    async def _kurtosis(self, values: np.ndarray) -> float:
        if len(values) < 4:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        normalized = (values - mean) / std
        return np.mean(normalized ** 4) - 3

    def register_custom_feature(self, name: str, feature_func: callable) -> None:
        self.feature_registry[name] = feature_func
        logger.info(f"Registered custom feature: {name}")

    @error_boundary(fallback={})
    async def apply_custom_features(self, df: pd.DataFrame) -> Dict[str, float]:
        features = {}
        
        for name, func in self.feature_registry.items():
            try:
                result = await func(df) if asyncio.iscoroutinefunction(func) else func(df)
                features[name] = result
            except Exception as e:
                logger.error(f"Error applying custom feature {name}: {e}")
                features[name] = 0.0
        
        return features

    async def get_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        importance_weights = {
            'rsi_14': 0.9,
            'volatility_20': 0.8,
            'sma_20': 0.7,
            'volume_ratio': 0.6,
            'bb_position': 0.8,
            'pe_ratio': 0.9,
            'debt_to_equity': 0.7
        }
        
        importance_scores = {}
        for feature_name, value in features.items():
            base_importance = importance_weights.get(feature_name, 0.5)
            
            value_significance = min(abs(value), 1.0) if value != 0 else 0.1
            importance_scores[feature_name] = base_importance * value_significance
        
        return importance_scores

    async def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        issues = []
        
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                issues.append(f"Feature {name} has invalid type: {type(value)}")
            elif np.isnan(value) or np.isinf(value):
                issues.append(f"Feature {name} has invalid value: {value}")
        
        return len(issues) == 0, issues