import asyncio
import pickle
from concurrent.futures import ThreadPoolExecutor, CancelledError
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, BatchNormalization, Dense, Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import redis

from .logger_config import logger
from .resource_manager import managed_tf_session
from .exceptions import ModelError
from .indicators.indicator_factory import IndicatorFactory

tf.get_logger().setLevel('ERROR')
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

except (RuntimeError, ValueError) as e:
    logger.warning(f"Could not configure TensorFlow devices: {e}")


class FeatureEngineer:
    def __init__(self):
        self.indicator_factory = IndicatorFactory()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            logger.warning("Empty dataframe provided to create_features")
            return pd.DataFrame()
        
        df = data.copy()
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        for name in self.indicator_factory.get_all_indicator_names():
            indicator = self.indicator_factory.create(name)
            if indicator is None:
                continue
            try:
                result = indicator.calculate(df)
                if result and hasattr(result, 'value'):
                    if isinstance(result.value, pd.Series):
                        df[name] = result.value
                    elif isinstance(result.value, np.ndarray):
                        df[name] = pd.Series(result.value, index=df.index)
                    elif isinstance(result.value, (int, float)):
                        df[name] = result.value
            except Exception as e:
                logger.debug(f"Failed to calculate indicator {name}: {e}")
                df[name] = np.nan
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        df.dropna(inplace=True)
        
        if self.feature_columns is None:
            self.feature_columns = df.columns.tolist()
        
        return df

    def scale_features(self, features: pd.DataFrame, fit: bool = False) -> Optional[np.ndarray]:
        if features is None or features.empty:
            logger.warning("Empty features provided to scale_features")
            return None
        
        features_clean = features.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.ffill().bfill()
        
        if features_clean.isnull().any().any():
            logger.warning("Features still contain NaN after cleaning")
            features_clean = features_clean.fillna(0)
        
        try:
            if fit:
                scaled = self.scaler.fit_transform(features_clean)
                self.feature_columns = features_clean.columns.tolist()
            else:
                if self.feature_columns is not None:
                    missing_cols = set(self.feature_columns) - set(features_clean.columns)
                    extra_cols = set(features_clean.columns) - set(self.feature_columns)
                    
                    if missing_cols or extra_cols:
                        logger.warning(f"Feature mismatch - Missing: {missing_cols}, Extra: {extra_cols}")
                        # Add missing columns with zero values
                        for col in missing_cols:
                            features_clean[col] = 0
                        # Reorder columns to match training feature order
                        features_clean = features_clean[self.feature_columns]
                else:
                    # If feature_columns is None, we cannot transform
                    logger.error("Cannot scale features: scaler not fitted (feature_columns is None)")
                    return None
                
                scaled = self.scaler.transform(features_clean)
            
            return scaled
        except Exception as e:
            logger.error(f"Error in scale_features: {e}")
            return None

    def inverse_scale_prediction(self, prediction: np.ndarray) -> np.ndarray:
        try:
            if self.scaler.n_features_in_ is None or self.scaler.n_features_in_ == 0:
                logger.warning("Scaler not fitted, returning prediction as-is")
                return prediction.flatten()
            
            dummy_array = np.zeros((len(prediction), self.scaler.n_features_in_))
            dummy_array[:, 0] = prediction.flatten()
            return self.scaler.inverse_transform(dummy_array)[:, 0]
        except Exception as e:
            logger.error(f"Error in inverse_scale_prediction: {e}")
            return prediction.flatten()

    def create_sequences(self, scaled_data: np.ndarray, target_data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        if scaled_data is None or len(scaled_data) == 0:
            return np.array([]), np.array([])
        
        if len(scaled_data) <= sequence_length:
            logger.warning(f"Not enough data for sequences: {len(scaled_data)} <= {sequence_length}")
            return np.array([]), np.array([])
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(target_data[i, 0])
        
        return np.array(X), np.array(y)


class BaseModel:
    def __init__(self, symbol: str, timeframe: str, model_path: str):
        self.model: Any = None
        self.is_trained = False
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.feature_engineer = FeatureEngineer()
        self.last_training_date = None
        self._is_closed = False
        self.logger = logger

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError(f"Model {self.symbol}-{self.timeframe} has been closed and cannot be used")

    def _get_model_paths(self, model_name: str, extension: str) -> Tuple[Path, Path]:
        safe_symbol = self.symbol.lower().replace('/', '').replace('\\', '')
        model_suffix = f"{safe_symbol}_{self.timeframe}"
        model_file = self.model_path / f"{model_name}_{model_suffix}.{extension}"
        scaler_file = self.model_path / f"scaler_{model_name}_{model_suffix}.pkl"
        return model_file, scaler_file

    def save_scaler(self, model_name: str):
        self._check_if_closed()
        _, scaler_file = self._get_model_paths(model_name, "")
        scaler_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.feature_engineer.scaler, f)
            self.logger.debug(f"Scaler saved to {scaler_file}")
        except Exception as e:
            self.logger.error(f"Error saving scaler: {e}")
            raise

    def load_scaler(self, model_name: str) -> bool:
        self._check_if_closed()
        _, scaler_file = self._get_model_paths(model_name, "")
        if scaler_file.exists():
            try:
                with open(scaler_file, 'rb') as f:
                    self.feature_engineer.scaler = pickle.load(f)
                self.logger.debug(f"Scaler loaded from {scaler_file}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading scaler: {e}")
                return False
        return False

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def cleanup(self):
        if self._is_closed:
            return
        self._is_closed = True
        try:
            if self.model:
                del self.model
                self.model = None
                self.logger.debug(f"Cleaned up model object for {self.symbol}-{self.timeframe}")
        except Exception as e:
            self.logger.error(f"Error during model cleanup for {self.symbol}-{self.timeframe}: {e}")

    def __del__(self):
        if not self._is_closed:
            self.cleanup()


class LSTMModel(BaseModel):
    def __init__(self, symbol: str, timeframe: str, model_path: str = "models/lstm", units=64, lr=0.001):
        super().__init__(symbol, timeframe, model_path)
        
        sequence_length_map = {
            "1h": 60, "4h": 48, "1d": 30, "1w": 24, "1M": 12
        }
        self.sequence_length = sequence_length_map.get(timeframe, 30)
        
        self.units = max(32, units // (2 if timeframe in ['1w', '1M'] else 1))
        self.lr = lr
        self.batch_size = 16 if timeframe in ['1w', '1M'] else 32
        self.input_shape: Optional[Tuple[int, int]] = None

    def _create_model(self):
        self._check_if_closed()
        if not self.input_shape:
            raise ModelError("Input shape must be set before creating LSTM model.")
        
        with managed_tf_session():
            self.model = Sequential([
                LSTM(self.units, input_shape=self.input_shape, return_sequences=True, 
                     dropout=0.2, recurrent_dropout=0.2),
                LSTM(self.units // 2, return_sequences=False, dropout=0.2),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(1, activation='linear')
            ])
            optimizer = Adam(learning_rate=self.lr, clipnorm=1.0)
            self.model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
            self.logger.debug(f"Created LSTM model for {self.symbol}-{self.timeframe}")

    def fit(self, data: pd.DataFrame, epochs=15, batch_size=None, validation_split=0.2) -> bool:
        self._check_if_closed()
        try:
            if data is None or data.empty or len(data) < self.sequence_length + 10:
                self.logger.warning(f"Insufficient data for {self.symbol}-{self.timeframe}, skipping training. Required: {self.sequence_length + 10}, Got: {len(data) if data is not None else 0}")
                return False
            
            features = self.feature_engineer.create_features(data)
            if features.empty or len(features) < self.sequence_length + 10:
                self.logger.warning(f"No features or insufficient features created for {self.symbol}-{self.timeframe}, skipping training.")
                return False
            
            scaled_features = self.feature_engineer.scale_features(features, fit=True)
            if scaled_features is None or len(scaled_features) == 0:
                self.logger.warning(f"Feature scaling failed for {self.symbol}-{self.timeframe}, skipping training.")
                return False

            self.input_shape = (self.sequence_length, scaled_features.shape[1])
            self._create_model()
            
            X, y = self.feature_engineer.create_sequences(scaled_features, scaled_features, self.sequence_length)
            if X.shape[0] == 0 or y.shape[0] == 0:
                self.logger.warning(f"Not enough data to create sequences for {self.symbol}-{self.timeframe}, skipping training.")
                return False

            effective_batch_size = min(batch_size or self.batch_size, X.shape[0] // 5)
            if effective_batch_size < 1:
                effective_batch_size = 1
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-7)
            ]
            
            with managed_tf_session():
                history = self.model.fit(X, y, epochs=epochs, batch_size=effective_batch_size,
                                        validation_split=validation_split, callbacks=callbacks,
                                        verbose=0, shuffle=False)

            final_val_loss = history.history.get('val_loss', [float('inf')])[-1]
            
            # Dynamic convergence threshold based on price standard deviation
            price_std = data['close'].std()
            # Use scaled threshold that's more lenient for high-volatility assets
            # The threshold is based on normalized loss (as features are scaled 0-1)
            # We use a factor of the variance to allow for reasonable error margins
            dynamic_threshold = max(0.01, min(0.5, price_std / data['close'].mean())) if price_std > 0 else 0.1

            if final_val_loss < dynamic_threshold:
                self.is_trained = True
                self.last_training_date = pd.Timestamp.now(tz='UTC')
                try:
                    self.save()
                    self.logger.info(f"LSTM model for {self.symbol}-{self.timeframe} trained and saved successfully with val_loss: {final_val_loss:.4f}")
                except Exception as save_e:
                    self.logger.error(f"Failed to save LSTM model for {self.symbol}-{self.timeframe}: {save_e}")
                    self.is_trained = False
                    return False
                return True
            else:
                self.logger.warning(f"LSTM training for {self.symbol}-{self.timeframe} did not converge. Final val_loss: {final_val_loss:.4f}")
                self.is_trained = False
                return False
        except Exception as e:
            self.logger.error(f"Error during LSTM training for {self.symbol}-{self.timeframe}: {e}", exc_info=True)
            self.is_trained = False
            return False

    def predict(self, data: pd.DataFrame) -> Optional[Tuple[np.ndarray, float]]:
        self._check_if_closed()
        if not self.model or not self.is_trained:
            self.logger.debug(f"Prediction skipped for {self.symbol}-{self.timeframe}: model not available or not trained.")
            return None
        
        try:
            if data is None or data.empty:
                self.logger.warning(f"Empty data for prediction {self.symbol}-{self.timeframe}")
                return None
            
            features = self.feature_engineer.create_features(data)
            if features.empty or len(features) < self.sequence_length:
                self.logger.warning(f"Not enough data for prediction for {self.symbol}-{self.timeframe}. Required: {self.sequence_length}, available: {len(features)}")
                return None

            last_sequence_features = features.tail(self.sequence_length)
            scaled_sequence = self.feature_engineer.scale_features(last_sequence_features, fit=False)
            if scaled_sequence is None or len(scaled_sequence) == 0:
                return None
            
            reshaped_sequence = scaled_sequence.reshape(1, self.sequence_length, scaled_sequence.shape[1])
            
            with managed_tf_session():
                scaled_prediction = self.model.predict(reshaped_sequence, verbose=0)

            if scaled_prediction is None or len(scaled_prediction) == 0:
                return None

            prediction = self.feature_engineer.inverse_scale_prediction(scaled_prediction)
            uncertainty = min(np.std(scaled_prediction) * 0.1, 0.5)
            
            return prediction, float(uncertainty)
        except Exception as e:
            self.logger.error(f"Error during LSTM prediction for {self.symbol}-{self.timeframe}: {e}", exc_info=True)
            return None

    def save(self):
        self._check_if_closed()
        if not self.model:
            raise ModelError("Cannot save a non-existent model.")
        try:
            model_file, _ = self._get_model_paths("lstm", "keras")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            
            with managed_tf_session():
                self.model.save(str(model_file))
            
            self.save_scaler("lstm")
            self.logger.info(f"LSTM model for {self.symbol}-{self.timeframe} saved to {model_file}")
        except Exception as e:
            self.logger.error(f"Error saving LSTM model for {self.symbol}-{self.timeframe}: {e}")
            raise

    def load(self):
        self._check_if_closed()
        try:
            model_file, _ = self._get_model_paths("lstm", "keras")
            if not model_file.exists():
                raise ModelError(f"Model file not found at {model_file}")
            
            if not self.load_scaler("lstm"):
                raise ModelError(f"Scaler not found for LSTM model {self.symbol}-{self.timeframe}. It will need retraining.")
            
            with managed_tf_session():
                self.model = tf.keras.models.load_model(str(model_file), compile=False)
                self.model.compile(optimizer=Adam(learning_rate=self.lr, clipnorm=1.0), loss='huber', metrics=['mae'])

            self.is_trained = True
            self.input_shape = self.model.layers[0].input_shape[1:]
            self.logger.info(f"LSTM model for {self.symbol}-{self.timeframe} loaded from {model_file}")
        except Exception as e:
            self.logger.error(f"Error loading LSTM model for {self.symbol}-{self.timeframe}: {e}")
            self.is_trained = False
            raise

    def cleanup(self):
        if self._is_closed:
            return
        super().cleanup()
        try:
            tf.keras.backend.clear_session()
            self.logger.debug(f"Keras session cleared for {self.symbol}-{self.timeframe}")
        except Exception as e:
            self.logger.error(f"Error clearing Keras session for {self.symbol}-{self.timeframe}: {e}")


class XGBoostModel(BaseModel):
    def __init__(self, symbol: str, timeframe: str, model_path: str = "models/xgboost", **params):
        super().__init__(symbol, timeframe, model_path)
        default_params = {
            "objective": "reg:squarederror", 
            "eval_metric": "rmse", 
            "n_estimators": 150,
            "learning_rate": 0.05, 
            "tree_method": "hist", 
            "max_depth": 5, 
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, data: pd.DataFrame, validation_split=0.2) -> bool:
        self._check_if_closed()
        try:
            if data is None or data.empty or len(data) < 50:
                self.logger.warning(f"Insufficient data for XGBoost {self.symbol}-{self.timeframe}")
                return False
            
            features = self.feature_engineer.create_features(data)
            if features.empty or len(features) < 50:
                self.logger.warning(f"No features created for XGBoost {self.symbol}-{self.timeframe}")
                return False
            
            target = features['close'].shift(-1)
            features = features[:-1]
            target = target[:-1].dropna()
            
            if len(features) != len(target) or len(features) == 0:
                self.logger.warning(f"Feature-target mismatch for XGBoost {self.symbol}-{self.timeframe}")
                return False

            split_index = max(1, int(len(features) * (1 - validation_split)))
            X_train, X_val = features[:split_index], features[split_index:]
            y_train, y_val = target[:split_index], target[split_index:]
            
            if len(X_train) == 0 or len(X_val) == 0:
                self.logger.warning(f"Insufficient split data for XGBoost {self.symbol}-{self.timeframe}")
                return False

            self.model.fit(X_train, y_train, 
                          eval_set=[(X_val, y_val)], 
                          verbose=False)
            
            self.is_trained = True
            self.last_training_date = pd.Timestamp.now(tz='UTC')
            
            try:
                self.save()
                self.logger.info(f"XGBoost model for {self.symbol}-{self.timeframe} trained and saved successfully")
            except Exception as save_e:
                self.logger.error(f"Failed to save XGBoost model for {self.symbol}-{self.timeframe}: {save_e}")
                self.is_trained = False
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error during XGBoost training for {self.symbol}-{self.timeframe}: {e}", exc_info=True)
            self.is_trained = False
            return False

    def predict(self, data: pd.DataFrame) -> Optional[Tuple[np.ndarray, float]]:
        self._check_if_closed()
        if not self.is_trained or self.model is None:
            self.logger.debug(f"Prediction skipped for XGBoost {self.symbol}-{self.timeframe}: model not trained")
            return None
        
        try:
            if data is None or data.empty:
                self.logger.warning(f"Empty data for XGBoost prediction {self.symbol}-{self.timeframe}")
                return None
            
            features = self.feature_engineer.create_features(data)
            if features.empty:
                self.logger.warning(f"No features for XGBoost prediction {self.symbol}-{self.timeframe}")
                return None
            
            last_features = features.tail(1)
            
            if last_features.empty:
                return None
            
            prediction = self.model.predict(last_features)
            
            uncertainty = 0.05
            if len(last_features) > 0:
                try:
                    predictions_sample = [self.model.predict(last_features)[0] for _ in range(5)]
                    uncertainty = min(np.std(predictions_sample), 0.5)
                except:
                    uncertainty = 0.05
            
            return prediction, float(uncertainty)
        except Exception as e:
            self.logger.error(f"Error during XGBoost prediction for {self.symbol}-{self.timeframe}: {e}", exc_info=True)
            return None

    def save(self):
        self._check_if_closed()
        try:
            model_file, _ = self._get_model_paths("xgboost", "json")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.model.save_model(str(model_file))
            self.logger.info(f"XGBoost model for {self.symbol}-{self.timeframe} saved to {model_file}")
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model for {self.symbol}-{self.timeframe}: {e}")
            raise

    def load(self):
        self._check_if_closed()
        try:
            model_file, _ = self._get_model_paths("xgboost", "json")
            if not model_file.exists():
                raise ModelError(f"Model file not found at {model_file}")
            
            self.model.load_model(str(model_file))
            self.is_trained = True
            self.logger.info(f"XGBoost model for {self.symbol}-{self.timeframe} loaded from {model_file}")
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model for {self.symbol}-{self.timeframe}: {e}")
            self.is_trained = False
            raise


class ModelManager:
    def __init__(self, model_path: str = 'models', redis_client: Optional[redis.Redis] = None):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.redis_client = redis_client
        self._cache: Dict[str, 'BaseModel'] = {}
        self._lock = asyncio.Lock()
        self._is_closed = False
        self.logger = logger
        self._training_executor = None
        self._prediction_executor = None

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("ModelManager has been closed and cannot be used")

    async def _get_executor(self, executor_type: str) -> ThreadPoolExecutor:
        self._check_if_closed()
        
        if executor_type == 'training':
            if self._training_executor is None or self._training_executor._shutdown:
                max_workers = max(1, (os.cpu_count() or 4) // 2)
                self._training_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='ModelTrainingExecutor')
            return self._training_executor
        elif executor_type == 'prediction':
            if self._prediction_executor is None or self._prediction_executor._shutdown:
                max_workers = max(1, (os.cpu_count() or 2))
                self._prediction_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='ModelPredictionExecutor')
            return self._prediction_executor
        else:
            raise ValueError("Invalid executor type specified.")

    async def _run_in_executor(self, func, *args, executor_type: str = 'prediction', **kwargs) -> Any:
        self._check_if_closed()
        loop = asyncio.get_running_loop()
        executor = await self._get_executor(executor_type)
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    def _get_model_class(self, model_type: str) -> Type['BaseModel']:
        if model_type == 'lstm':
            return LSTMModel
        if model_type == 'xgboost':
            return XGBoostModel
        raise ValueError(f"Unknown model type: {model_type}")

    async def get_model(self, model_type: str, symbol: str, timeframe: str, auto_load: bool = True) -> Optional['BaseModel']:
        self._check_if_closed()
        key = f"{model_type}-{symbol}-{timeframe}"
        
        async with self._lock:
            if key in self._cache:
                cached_model = self._cache[key]
                if not cached_model._is_closed:
                    return cached_model
                del self._cache[key]
        
        model_dir = self.model_path / model_type
        model_class = self._get_model_class(model_type)
        model_path_str = str(model_dir)

        try:
            model = await self._run_in_executor(lambda: model_class(symbol=symbol, timeframe=timeframe, model_path=model_path_str), executor_type='prediction')
            
            if auto_load:
                if model_type == 'lstm':
                    model_file, scaler_file = model._get_model_paths(model_type, "keras")
                else:
                    model_file, scaler_file = model._get_model_paths(model_type, "json")
                
                if model_file.exists() and scaler_file.exists():
                    try:
                        await self._run_in_executor(model.load, executor_type='prediction')
                        self.logger.info(f"Loaded existing {model_type.upper()} model for {key}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_type.upper()} model for {key}: {e}")
                        model.is_trained = False
                else:
                    self.logger.info(f"No existing {model_type.upper()} model found for {key}, needs training")
                    model.is_trained = False

            async with self._lock:
                if key not in self._cache:
                    self._cache[key] = model
                else:
                    model.cleanup()
                    return self._cache[key]
            return model
        except Exception as e:
            self.logger.error(f"Could not get or create {model_type.upper()} model for {key}: {e}", exc_info=True)
            return None

    async def train_model(self, model_type: str, symbol: str, timeframe: str, data: pd.DataFrame, **kwargs) -> bool:
        self._check_if_closed()
        
        if data is None or data.empty:
            self.logger.warning(f"Cannot train {model_type} for {symbol}-{timeframe}: empty data")
            return False
        
        model = await self.get_model(model_type, symbol, timeframe, auto_load=False)
        if not model:
            self.logger.error(f"Failed to create model {model_type} for {symbol}-{timeframe}")
            return False
        
        try:
            success = await self._run_in_executor(model.fit, data, executor_type='training', **kwargs)
            if success:
                self.logger.info(f"Successfully trained {model_type.upper()} model for {symbol}-{timeframe}")
                async with self._lock:
                    key = f"{model_type}-{symbol}-{timeframe}"
                    self._cache[key] = model
                return True
            else:
                self.logger.warning(f"Training failed for {model_type.upper()} model {symbol}-{timeframe}")
                return False
        except Exception as e:
            self.logger.error(f"Error training {model_type.upper()} model for {symbol}-{timeframe}: {e}", exc_info=True)
            return False

    async def initialize_all_models(self):
        self._check_if_closed()
        self.logger.info("Initializing all models (LSTM, XGBoost)...")
        tasks = []
        model_files = []
        
        for model_type_dir in self.model_path.iterdir():
            if not model_type_dir.is_dir():
                continue
            
            model_type = model_type_dir.name
            if model_type == "lstm": 
                ext = "keras"
            elif model_type == "xgboost": 
                ext = "json"
            else: 
                continue
            
            for file in model_type_dir.glob(f"*.{ext}"):
                model_files.append((file, model_type))

        for file, model_type in model_files:
            try:
                base_name = file.stem
                if model_type == 'lstm': 
                    parts_str = base_name.replace("lstm_", "")
                elif model_type == 'xgboost': 
                    parts_str = base_name.replace("xgboost_", "")
                else: 
                    continue
                
                symbol, timeframe = parts_str.rsplit('_', 1)
                symbol_formatted = symbol.upper()
                if '/' not in symbol_formatted and len(symbol_formatted) > 4:
                    symbol_formatted = symbol_formatted.replace("USDT", "/USDT", 1)

                tasks.append(self.get_model(model_type, symbol_formatted, timeframe))
            except (IndexError, ValueError) as e:
                self.logger.warning(f"Could not parse model file name: {file.name}. Error: {e}")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Error during model initialization: {res}", exc_info=res)
        self.logger.info(f"Model initialization finished. Loaded/Created {len(tasks)} models.")

    async def predict_with_confidence(self, model_type: str, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        self._check_if_closed()
        
        if data is None or data.empty:
            self.logger.warning(f"Empty data for prediction {model_type} on {symbol}-{timeframe}")
            return None
        
        model = await self.get_model(model_type, symbol, timeframe)
        if not model or not model.is_trained:
            self.logger.info(f"Prediction for {model_type} on {symbol}-{timeframe} skipped: model not trained or available.")
            return None
        
        try:
            prediction_result = await self._run_in_executor(model.predict, data, executor_type='prediction')
            
            # Validate prediction_result structure
            if prediction_result is None:
                self.logger.debug(f"Prediction returned None for {model_type} on {symbol}-{timeframe}")
                return None
            
            if not isinstance(prediction_result, tuple) or len(prediction_result) != 2:
                self.logger.warning(f"Invalid prediction result format for {model_type} on {symbol}-{timeframe}: expected tuple of length 2, got {type(prediction_result)}")
                return None
            
            prediction, uncertainty = prediction_result
            
            # Validate prediction array
            if prediction is None:
                self.logger.debug(f"Prediction array is None for {model_type} on {symbol}-{timeframe}")
                return None
                
            if not hasattr(prediction, '__len__') or len(prediction) == 0:
                self.logger.warning(f"Prediction array is empty for {model_type} on {symbol}-{timeframe}")
                return None
            
            raw_confidence = 50.0
            calibrated_confidence = 50.0
            
            current_price = data['close'].iloc[-1]
            if current_price > 0 and not np.isnan(prediction[0]) and not np.isinf(prediction[0]):
                price_change_pct = abs(prediction[0] - current_price) / current_price * 100
                raw_confidence = min(price_change_pct * 10, 100.0)
                calibrated_confidence = raw_confidence
            
            final_confidence = max(0.0, min(100.0, calibrated_confidence * (1 - uncertainty ** 2)))
            
            return {
                'prediction': float(prediction[0]),
                'confidence': final_confidence,
                'raw_confidence': raw_confidence,
                'uncertainty': uncertainty
            }
        except (RuntimeError, CancelledError) as e:
            self.logger.warning(f"Prediction task cancelled or failed for {model_type} on {symbol}-{timeframe}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_type} on {symbol}-{timeframe}: {e}")
            return None

    async def record_signal_performance(self, model_type: str, symbol: str, timeframe: str, success: bool):
        self._check_if_closed()
        model = await self.get_model(model_type, symbol, timeframe)
        if model and hasattr(model, 'calibrator') and model.calibrator:
            confidence = 0.5
            model.calibrator.add_prediction(f"{model_type}_{symbol}_{timeframe}", confidence, success)

    async def shutdown(self):
        if self._is_closed:
            return
        
        self._is_closed = True
        self.logger.info("Shutting down ModelManager...")
        
        async with self._lock:
            models_to_clean = list(self._cache.values())
            self._cache.clear()
        
        for model in models_to_clean:
            try:
                model.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up model {model.symbol}-{model.timeframe}: {e}")
        
        for executor_attr in ['_training_executor', '_prediction_executor']:
            if hasattr(self, executor_attr):
                executor = getattr(self, executor_attr)
                if executor:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except Exception as e:
                        self.logger.error(f"Error shutting down executor: {e}")
        
        self.logger.info("ModelManager shut down.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
        return False