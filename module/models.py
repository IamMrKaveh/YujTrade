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
import redis.asyncio as redis

from module.logger_config import logger
from module.resource_manager import managed_tf_session
from module.exceptions import ModelError
from module.indicators import TechnicalIndicator, get_all_indicators

tf.get_logger().setLevel('ERROR')
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        tf.config.set_visible_devices([], 'GPU')
except (RuntimeError, ValueError) as e:
    logger.warning(f"Could not configure TensorFlow devices: {e}")


class FeatureEngineer:
    def __init__(self, indicators):
        self.indicators = indicators
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for name, indicator in self.indicators.items():
            try:
                result = indicator.calculate(df)
                if result and hasattr(result, 'value'):
                    df[name] = result.value
            except Exception:
                df[name] = np.nan
        
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.dropna(inplace=True)
        return df

    def scale_features(self, features: pd.DataFrame, fit: bool = False) -> Optional[np.ndarray]:
        if features.empty:
            return None
        if fit:
            return self.scaler.fit_transform(features)
        return self.scaler.transform(features)

    def inverse_scale_prediction(self, prediction: np.ndarray) -> np.ndarray:
        dummy_array = np.zeros((len(prediction), self.scaler.n_features_in_))
        dummy_array[:, 0] = prediction.flatten()
        return self.scaler.inverse_transform(dummy_array)[:, 0]

    def create_sequences(self, scaled_data: np.ndarray, target_data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
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
        self.logger = logger
        self.feature_engineer = FeatureEngineer(get_all_indicators())
        self._is_closed = False

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError(f"Model {self.symbol}-{self.timeframe} has been closed and cannot be used")

    def _get_model_paths(self, model_name: str, extension: str) -> Tuple[Path, Path]:
        model_suffix = f"_{self.symbol.lower().replace('/', '')}-{self.timeframe}"
        model_file = self.model_path / f"{model_name}{model_suffix}.{extension}"
        scaler_file = self.model_path / f"scaler_{model_name}{model_suffix}.pkl"
        return model_file, scaler_file

    def save_scaler(self, model_name: str):
        self._check_if_closed()
        _, scaler_file = self._get_model_paths(model_name, "")
        scaler_file.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.feature_engineer.scaler, f)
        logger.debug(f"Scaler saved to {scaler_file}")

    def load_scaler(self, model_name: str) -> bool:
        self._check_if_closed()
        _, scaler_file = self._get_model_paths(model_name, "")
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.feature_engineer.scaler = pickle.load(f)
            logger.debug(f"Scaler loaded from {scaler_file}")
            return True
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
            "1h": 60,
            "4h": 48,
            "1d": 30,
            "1w": 24,
            "1M": 12
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
            self.logger.debug(f"Created optimized LSTM model for {self.symbol}-{self.timeframe}.")

    def fit(self, data: pd.DataFrame, epochs=15, batch_size=None, validation_split=0.2) -> bool:
        self._check_if_closed()
        try:
            features = self.feature_engineer.create_features(data)
            if features.empty:
                self.logger.warning(f"No features created for {self.symbol}-{self.timeframe}, skipping training.")
                return False
            
            scaled_features = self.feature_engineer.scale_features(features, fit=True)
            if scaled_features is None:
                self.logger.warning(f"Feature scaling failed for {self.symbol}-{self.timeframe}, skipping training.")
                return False

            self.input_shape = (self.sequence_length, scaled_features.shape[1])
            self._create_model()
            
            X, y = self.feature_engineer.create_sequences(scaled_features, scaled_features, self.sequence_length)
            if X.shape[0] == 0:
                self.logger.warning(f"Not enough data to create sequences for {self.symbol}-{self.timeframe}, skipping training.")
                return False

            effective_batch_size = batch_size or self.batch_size
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-7)
            ]
            
            with managed_tf_session():
                history = self.model.fit(X, y, epochs=epochs, batch_size=effective_batch_size,
                                        validation_split=validation_split, callbacks=callbacks,
                                        verbose=0, shuffle=False)

            final_val_loss = history.history.get('val_loss', [float('inf')])[-1]
            if final_val_loss < 1.0:
                self.is_trained = True
                self.logger.info(f"LSTM model for {self.symbol}-{self.timeframe} trained successfully with val_loss: {final_val_loss:.4f}")
                return True
            else:
                self.logger.warning(f"LSTM training for {self.symbol}-{self.timeframe} did not converge. Final val_loss: {final_val_loss:.4f}")
                self.is_trained = False
                return False
        except Exception as e:
            self.logger.error(f"Error during LSTM training for {self.symbol}-{self.timeframe}: {e}", exc_info=True)
            self.is_trained = False
            return False

    def predict(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_if_closed()
        if not self.model or not self.is_trained:
            self.logger.debug(f"Prediction skipped for {self.symbol}-{self.timeframe}: model not available or not trained.")
            return None
        
        try:
            features = self.feature_engineer.create_features(data)
            if len(features) < self.sequence_length:
                self.logger.warning(f"Not enough data for prediction for {self.symbol}-{self.timeframe}. Required: {self.sequence_length}, available: {len(features)}")
                return None

            last_sequence_features = features.tail(self.sequence_length)
            scaled_sequence = self.feature_engineer.scale_features(last_sequence_features, fit=False)
            if scaled_sequence is None:
                return None
            
            reshaped_sequence = scaled_sequence.reshape(1, self.sequence_length, scaled_sequence.shape[1])
            
            with managed_tf_session():
                scaled_prediction = self.model.predict(reshaped_sequence, verbose=0)

            if scaled_prediction is None:
                return None

            return self.feature_engineer.inverse_scale_prediction(scaled_prediction)
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
                raise ModelError(f"Scaler not found for LSTM model {self.symbol}-{self.timeframe}.")
            
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
        super().cleanup()
        try:
            tf.keras.backend.clear_session()
            self.logger.debug(f"Keras session cleared for {self.symbol}-{self.timeframe}")
        except Exception as e:
            self.logger.error(f"Error clearing Keras session for {self.symbol}-{self.timeframe}: {e}")


class XGBoostModel(BaseModel):
    def __init__(self, symbol: str, timeframe: str, model_path: str = "models/xgboost", **params):
        super().__init__(symbol, timeframe, model_path)
        self.params = params or {
            "objective": "reg:squarederror", 
            "eval_metric": "rmse", 
            "n_estimators": 150,
            "learning_rate": 0.05, 
            "tree_method": "hist", 
            "max_depth": 5, 
            "subsample": 0.8,
        }
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, data: pd.DataFrame, validation_split=0.2):
        self._check_if_closed()
        try:
            features = self.feature_engineer.create_features(data)
            if features.empty:
                return False
            
            target = features['close'].shift(-1).dropna()
            features = features.iloc[:-1]

            split_index = int(len(features) * (1 - validation_split))
            X_train, X_val = features[:split_index], features[split_index:]
            y_train, y_val = target[:split_index], target[split_index:]

            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            self.is_trained = True
            return True
        except Exception as e:
            self.logger.error(f"Error during XGBoost training: {e}")
            return False

    def predict(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_if_closed()
        if not self.is_trained:
            return None
        
        try:
            features = self.feature_engineer.create_features(data)
            if features.empty:
                return None
            
            last_features = features.tail(1)
            return self.model.predict(last_features)
        except Exception as e:
            self.logger.error(f"Error during XGBoost prediction: {e}")
            return None

    def save(self):
        self._check_if_closed()
        try:
            model_file, _ = self._get_model_paths("xgboost", "json")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.model.save_model(str(model_file))
            self.logger.info(f"XGBoost model for {self.symbol}-{self.timeframe} saved to {model_file}")
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {e}")
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
            self.logger.error(f"Error loading XGBoost model: {e}")
            raise


class ModelManager:
    def __init__(self, model_path: str = 'models', redis_client: Optional[redis.Redis] = None):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.redis_client = redis_client
        self._cache: Dict[str, 'BaseModel'] = {}
        self._lock = asyncio.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = asyncio.Lock()
        self._is_closed = False
        self.logger = logger

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("ModelManager has been closed and cannot be used")

    async def _get_executor(self) -> ThreadPoolExecutor:
        self._check_if_closed()
        async with self._executor_lock:
            if self._executor is None or self._executor._shutdown:
                max_workers = max(1, (os.cpu_count() or 1) - 1)
                self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='ModelExecutor')
            return self._executor

    async def _run_in_executor(self, func, *args, **kwargs) -> Any:
        self._check_if_closed()
        loop = asyncio.get_running_loop()
        executor = await self._get_executor()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    def _get_model_class(self, model_type: str) -> Type['BaseModel']:
        if model_type == 'lstm':
            return LSTMModel
        if model_type == 'xgboost':
            return XGBoostModel
        raise ValueError(f"Unknown model type: {model_type}")

    async def get_model(self, model_type: str, symbol: str, timeframe: str) -> Optional['BaseModel']:
        self._check_if_closed()
        key = f"{model_type}-{symbol}-{timeframe}"
        
        async with self._lock:
            if key in self._cache:
                return self._cache[key]
        
        model_dir = self.model_path / model_type
        model_class = self._get_model_class(model_type)
        model_path_str = str(model_dir)

        try:
            model = await self._run_in_executor(model_class, symbol=symbol, timeframe=timeframe, model_path=model_path_str)
            
            model_file, _ = model._get_model_paths(model_type, "keras" if model_type == 'lstm' else 'json')
            
            if model_file.exists():
                try:
                    await self._run_in_executor(model.load)
                    self.logger.info(f"Loaded existing {model_type.upper()} model for {key}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_type.upper()} model for {key}, it will need training: {e}")
            else:
                self.logger.info(f"No existing {model_type.upper()} model found for {key}. A new one will be trained.")

            async with self._lock:
                self._cache[key] = model
            return model
        except Exception as e:
            self.logger.error(f"Could not get or create {model_type.upper()} model for {key}: {e}", exc_info=True)
            return None

    async def initialize_all_models(self):
        self._check_if_closed()
        self.logger.info("Initializing all models (LSTM, XGBoost)...")
        tasks = []
        model_files = []
        for model_type_dir in self.model_path.iterdir():
            if model_type_dir.is_dir():
                model_type = model_type_dir.name
                if model_type == "lstm":
                    ext = "keras"
                elif model_type == "xgboost":
                    ext = "json"
                else:
                    continue
                
                for file in model_type_dir.glob(f"*{ext}"):
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
                
                symbol, timeframe = parts_str.split('-', 1)
                symbol = symbol.upper().replace('', '/', 1) if '/' not in symbol.upper() else symbol.upper()

                tasks.append(self.get_model(model_type, symbol, timeframe))
            except (IndexError, ValueError) as e:
                self.logger.warning(f"Could not parse model file name: {file.name}. Error: {e}")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Error during model initialization: {res}", exc_info=res)
        self.logger.info(f"Model initialization finished. Loaded/Created {len(tasks)} models.")

    async def predict(self, model_type: str, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[Any]:
        self._check_if_closed()
        model = await self.get_model(model_type, symbol, timeframe)
        if not model or not model.is_trained:
            return None
        
        try:
            prediction = await self._run_in_executor(model.predict, data)
            return prediction
        except (RuntimeError, CancelledError) as e:
            self.logger.warning(f"Prediction task cancelled or failed for {model_type} on {symbol}-{timeframe}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_type} on {symbol}-{timeframe}: {e}")
            return None

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
        
        async with self._executor_lock:
            executor = self._executor
            if executor:
                self._executor = None
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                    self.logger.info("ThreadPoolExecutor shut down successfully.")
                except Exception as e:
                    self.logger.error(f"Error shutting down executor: {e}")
        
        self.logger.info("ModelManager shut down.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
        return False