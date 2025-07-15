from typing import Dict, List, Any, Optional, Tuple, Union, Type
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
import joblib
import json
from pathlib import Path

from ..processors.error_handler import error_boundary, DataProcessingError
from ..processors.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"


class ModelStatus(Enum):
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    ERROR = "error"


@dataclass
class ModelConfig:
    model_type: ModelType
    target_column: str
    feature_columns: List[str] = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    enable_feature_selection: bool = True
    enable_hyperparameter_tuning: bool = False
    scoring_metric: str = "accuracy"
    max_training_time: int = 300


@dataclass
class ModelMetrics:
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None


@dataclass
class ModelInfo:
    model_id: str
    name: str
    type: ModelType
    status: ModelStatus
    created_at: datetime = field(default_factory=datetime.now)
    trained_at: Optional[datetime] = None
    version: str = "1.0.0"
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    feature_count: int = 0
    training_samples: int = 0


class ModelEngine:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model = None
        self.preprocessor = None
        self.feature_engineer = FeatureEngineer()
        self.model_info = ModelInfo(
            model_id=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"{model_config.model_type.value}_model",
            type=model_config.model_type,
            status=ModelStatus.UNTRAINED
        )
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            if self.config.model_type == ModelType.CLASSIFICATION:
                self._initialize_classification_models()
            elif self.config.model_type == ModelType.REGRESSION:
                self._initialize_regression_models()
            elif self.config.model_type == ModelType.TIME_SERIES:
                self._initialize_time_series_models()
            elif self.config.model_type == ModelType.CLUSTERING:
                self._initialize_clustering_models()
        except ImportError as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _initialize_classification_models(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            ),
            'svm': SVC(
                random_state=self.config.random_state,
                probability=True
            )
        }
        
        self.model = self.models['random_forest']

    def _initialize_regression_models(self) -> None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.svm import SVR
        
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(
                random_state=self.config.random_state
            ),
            'svr': SVR()
        }
        
        self.model = self.models['random_forest']

    def _initialize_time_series_models(self) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            
            self.models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state,
                    n_jobs=-1
                ),
                'linear_regression': LinearRegression()
            }
            
            self.model = self.models['random_forest']
            
        except ImportError:
            logger.warning("Advanced time series libraries not available, using basic models")
            self._initialize_regression_models()

    def _initialize_clustering_models(self) -> None:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.mixture import GaussianMixture
        
        self.models = {
            'kmeans': KMeans(
                n_clusters=3,
                random_state=self.config.random_state,
                n_init=10
            ),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'gaussian_mixture': GaussianMixture(
                n_components=3,
                random_state=self.config.random_state
            )
        }
        
        self.model = self.models['kmeans']

    @error_boundary(fallback=None)
    async def train_model(self, training_data: Union[pd.DataFrame, Dict, List[Dict]]) -> Dict[str, Any]:
        self.model_info.status = ModelStatus.TRAINING
        start_time = datetime.now()
        
        try:
            df = await self._prepare_training_data(training_data)
            X, y = await self._prepare_features_and_target(df)
            
            if self.config.enable_feature_selection:
                X = await self._select_features(X, y)
            
            self._setup_preprocessor(X)
            X_processed = self.preprocessor.fit_transform(X)
            
            if self.config.enable_hyperparameter_tuning:
                self.model = await self._tune_hyperparameters(X_processed, y)
            
            self.model.fit(X_processed, y)
            
            training_time = (datetime.now() - start_time).total_seconds()
            self.model_info.metrics.training_time = training_time
            self.model_info.trained_at = datetime.now()
            self.model_info.status = ModelStatus.TRAINED
            self.model_info.feature_count = X_processed.shape[1]
            self.model_info.training_samples = X_processed.shape[0]
            
            metrics = await self._evaluate_model(X_processed, y)
            self.model_info.metrics = metrics
            
            return {
                "status": "success",
                "model_id": self.model_info.model_id,
                "training_time": training_time,
                "metrics": metrics.__dict__,
                "feature_count": self.model_info.feature_count,
                "training_samples": self.model_info.training_samples
            }
            
        except Exception as e:
            self.model_info.status = ModelStatus.ERROR
            logger.error(f"Training failed: {e}")
            raise DataProcessingError(f"Model training failed: {e}")

    async def _prepare_training_data(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise DataProcessingError(f"Unsupported data type: {type(data)}")

    async def _prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if self.config.target_column not in df.columns:
            raise DataProcessingError(f"Target column '{self.config.target_column}' not found in data")
        
        y = df[self.config.target_column]
        
        if self.config.feature_columns:
            missing_features = [col for col in self.config.feature_columns if col not in df.columns]
            if missing_features:
                logger.warning(f"Missing feature columns: {missing_features}")
            X = df[self.config.feature_columns].select_dtypes(include=[np.number])
        else:
            X = df.select_dtypes(include=[np.number]).drop(columns=[self.config.target_column])
        
        if X.empty:
            raise DataProcessingError("No numeric features found for training")
        
        return X, y

    async def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        try:
            if self.config.model_type in [ModelType.CLASSIFICATION, ModelType.REGRESSION]:
                from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                
                if self.config.model_type == ModelType.CLASSIFICATION:
                    selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
                else:
                    selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
                
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()]
                logger.info(f"Selected {len(selected_features)} features: {list(selected_features)}")
                
                return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        except ImportError:
            logger.warning("Feature selection not available, using all features")
        
        return X

    def _setup_preprocessor(self, X: pd.DataFrame) -> None:
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        self.preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    async def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        try:
            from sklearn.model_selection import GridSearchCV
            
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                }
            }
            
            model_name = type(self.model).__name__.lower()
            if 'randomforest' in model_name:
                param_grid = param_grids['random_forest']
            elif 'logistic' in model_name:
                param_grid = param_grids['logistic_regression']
            else:
                return self.model
            
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=self.config.cross_validation_folds,
                scoring=self.config.scoring_metric,
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
            
        except ImportError:
            logger.warning("GridSearchCV not available, skipping hyperparameter tuning")
            return self.model

    @error_boundary(fallback=ModelMetrics())
    async def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = ModelMetrics()
        
        if self.config.model_type == ModelType.CLASSIFICATION:
            cv_scores = cross_val_score(self.model, X, y, cv=self.config.cross_validation_folds, scoring='accuracy')
            metrics.accuracy = np.mean(cv_scores)
            
            y_pred = self.model.predict(X)
            metrics.precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            metrics.recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            metrics.f1_score = f1_score(y, y_pred, average='weighted', zero_division=0)
            
        elif self.config.model_type == ModelType.REGRESSION:
            y_pred = self.model.predict(X)
            metrics.mse = mean_squared_error(y, y_pred)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(y, y_pred)
            metrics.r2_score = r2_score(y, y_pred)
        
        return metrics

    @error_boundary(fallback=None)
    async def predict(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> Dict[str, Any]:
        if self.model_info.status != ModelStatus.TRAINED:
            raise DataProcessingError("Model must be trained before making predictions")
        
        start_time = datetime.now()
        
        try:
            df = await self._prepare_prediction_data(input_data)
            X = self._prepare_prediction_features(df)
            X_processed = self.preprocessor.transform(X)
            
            predictions = self.model.predict(X_processed)
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            self.model_info.metrics.prediction_time = prediction_time
            
            result = {
                "status": "success",
                "predictions": predictions.tolist(),
                "prediction_time": prediction_time,
                "model_id": self.model_info.model_id,
                "prediction_count": len(predictions)
            }
            
            if hasattr(self.model, 'predict_proba') and self.config.model_type == ModelType.CLASSIFICATION:
                probabilities = self.model.predict_proba(X_processed)
                result["probabilities"] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise DataProcessingError(f"Prediction failed: {e}")

    async def _prepare_prediction_data(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        return await self._prepare_training_data(data)

    def _prepare_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.feature_columns:
            available_features = [col for col in self.config.feature_columns if col in df.columns]
            if not available_features:
                raise DataProcessingError("No required features found in prediction data")
            return df[available_features].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            if self.config.target_column in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[self.config.target_column])
            return numeric_df

    async def save_model(self, file_path: str) -> bool:
        try:
            model_data = {
                'model': self.model,
                'preprocessor': self.preprocessor,
                'config': self.config,
                'model_info': self.model_info
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"Model saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    @classmethod
    async def load_model(cls, file_path: str) -> 'ModelEngine':
        try:
            model_data = joblib.load(file_path)
            
            engine = cls(model_data['config'])
            engine.model = model_data['model']
            engine.preprocessor = model_data['preprocessor']
            engine.model_info = model_data['model_info']
            
            logger.info(f"Model loaded from {file_path}")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_info.model_id,
            "name": self.model_info.name,
            "type": self.model_info.type.value,
            "status": self.model_info.status.value,
            "created_at": self.model_info.created_at.isoformat(),
            "trained_at": self.model_info.trained_at.isoformat() if self.model_info.trained_at else None,
            "version": self.model_info.version,
            "metrics": self.model_info.metrics.__dict__,
            "feature_count": self.model_info.feature_count,
            "training_samples": self.model_info.training_samples
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        try:
            if hasattr(self.preprocessor, 'named_steps'):
                feature_names = self.preprocessor.named_steps.get('selector', None)
                if feature_names and hasattr(feature_names, 'get_feature_names_out'):
                    names = feature_names.get_feature_names_out()
                else:
                    names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            else:
                names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
            importance_dict = dict(zip(names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None