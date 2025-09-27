import asyncio
import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, BatchNormalization, Dense, Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas_ta as ta
import os

from module.logger_config import logger

tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')

class LSTMModel:
    def __init__(self, input_shape=(60, 17), units=64, lr=0.001, model_path='lstm-model', symbol=None, timeframe=None):
        self.model = None
        self.input_shape = input_shape
        self.trained = False
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.logger = logger
        self._setup_gpu()
        self._load_or_create_model(units, lr)
        
    def clear_cache(self):
        try:
            if hasattr(self, 'model') and self.model:
                del self.model
                self.model = None
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
                self.logger.info(f"Cleared cache for model {self.symbol}-{self.timeframe}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error during model cleanup for {self.symbol}-{self.timeframe}: {e}")

    def _setup_gpu(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    def _get_model_paths(self):
        model_suffix = ""
        if self.symbol and self.timeframe:
            model_suffix = f"_{self.symbol.lower().replace('/', '')}-{self.timeframe}"
        elif self.symbol:
            model_suffix = f"_{self.symbol.lower().replace('/', '')}"
            
        model_file = self.model_path / f"model{model_suffix}.keras"
        scaler_file = self.model_path / f"scaler{model_suffix}.pkl"
        return model_file, scaler_file

    def _load_or_create_model(self, units, lr):
        model_file, scaler_file = self._get_model_paths()
        
        if model_file.exists() and scaler_file.exists():
            try:
                self.model = tf.keras.models.load_model(str(model_file), compile=False)
                self.model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), 
                                    loss='huber', metrics=['mae'])
                
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.trained = True
                self.is_fitted = True
                self.logger.info(f"Loaded existing model and scaler from {model_file}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load model {model_file}: {e}. Creating a new one.")
        
        self._create_model(units, lr)

    def _create_model(self, units, lr):
        try:
            self.model = Sequential([
                LSTM(units, input_shape=self.input_shape, return_sequences=True, 
                    dropout=0.15, recurrent_dropout=0.15),
                LSTM(units // 2, return_sequences=True, 
                    dropout=0.15, recurrent_dropout=0.15),
                LSTM(units // 4, return_sequences=False, 
                    dropout=0.1, recurrent_dropout=0.1),
                Dropout(0.25),
                Dense(50, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='linear')
            ])
            optimizer = Adam(learning_rate=lr, clipnorm=1.0)
            self.model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
            self.logger.info("Created a new LSTM model instance.")
        except Exception as e:
            self.logger.error(f"Failed to create LSTM model: {e}")
            self.model = None

    def save_model(self):
        if not self.model or not self.trained or not self.is_fitted:
            return False
        
        try:
            model_file, scaler_file = self._get_model_paths()
            self.model.save(str(model_file))
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.logger.info(f"Saved model to {model_file} and scaler to {scaler_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < 50:
            return pd.DataFrame()
        
        df = data.copy()
        
        try:
            df.ta.strategy("common", append=True)
            
            feature_cols = [col for col in df.columns if col.startswith(('RSI', 'MACD', 'BB', 'STOCH', 'ATR', 'SMA', 'EMA'))]
            if not feature_cols:
                return pd.DataFrame()
                
            features_df = df[feature_cols].copy()
            features_df['close_norm'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            features_df['volume_norm'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            features_df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            features_df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
            features_df['price_change_1'] = df['close'].pct_change(1)
            features_df['price_change_5'] = df['close'].pct_change(5)
            features_df['volume_change'] = df['volume'].pct_change(1)

        except Exception as e:
            self.logger.warning(f"Feature creation failed: {e}. Returning empty dataframe.")
            return pd.DataFrame()
        
        features_df.fillna(method='ffill', inplace=True)
        features_df.fillna(method='bfill', inplace=True)
        features_df.fillna(0, inplace=True)
        features_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        current_num_features = features_df.shape[1]
        target_num_features = self.input_shape[1]

        if current_num_features > target_num_features:
            features_df = features_df.iloc[:, :target_num_features]
        elif current_num_features < target_num_features:
            padding_cols = target_num_features - current_num_features
            padding_df = pd.DataFrame(0, index=features_df.index, columns=[f'pad_{i}' for i in range(padding_cols)])
            features_df = pd.concat([features_df, padding_df], axis=1)

        return features_df

    def _validate_dataframe(self, data):
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        if not all(col in data.columns for col in required_columns):
            missing = set(required_columns) - set(data.columns)
            raise ValueError(f"Missing one or more required columns: {missing}")
        return True

    def prepare_sequences(self, data, window=None, for_training=True):
        try:
            window = window or self.input_shape[0]
            self._validate_dataframe(data)
            features_df = self._create_features(data)
            if features_df.empty or len(features_df) < window:
                return np.array([]), np.array([])
            
            feature_values = features_df.values
            target_values = data['close'].values

            if for_training:
                if len(target_values) > 0:
                    self.scaler.fit(target_values.reshape(-1, 1))
                    self.is_fitted = True
            
            if not self.is_fitted: return np.array([]), np.array([])

            target_scaled = self.scaler.transform(target_values.reshape(-1, 1)).flatten()
            
            X, y = [], []
            for i in range(window, len(feature_values)):
                X.append(feature_values[i-window:i])
                y.append(target_scaled[i])
            
            if not X or not y:
                return np.array([]), np.array([])

            X, y = np.array(X), np.array(y)
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return X, y
        except Exception as e:
            self.logger.error(f"Error in prepare_sequences: {e}")
            return np.array([]), np.array([])

    def fit(self, data, epochs=15, batch_size=32, verbose=0, validation_split=0.2):
        if self.trained or not self.model: return self.trained
        
        X, y = self.prepare_sequences(data, for_training=True)
        if X.size == 0 or y.size == 0: return False
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-7)
        ]
        
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks,
                                verbose=verbose, shuffle=False)
        
        final_val_loss = history.history.get('val_loss', [float('inf')])[-1]
        if final_val_loss < 1.0:
            self.trained = True
            self.save_model()
            return True
        return False

    def predict(self, data):
        if not self.is_ready(): return None
        
        try:
            self._validate_dataframe(data)
            features_df = self._create_features(data)
            if features_df.empty or len(features_df) < self.input_shape[0]: return None
            
            last_sequence = features_df.tail(self.input_shape[0]).values
            last_sequence = np.nan_to_num(last_sequence, nan=0.0, posinf=1.0, neginf=-1.0)
            
            X = last_sequence.reshape(1, self.input_shape[0], self.input_shape[1])
            
            prediction = self.model.predict(X, verbose=0)
            return self.scaler.inverse_transform(prediction).flatten()
        except Exception as e:
            self.logger.error(f"Error in predict: {e}")
            return None

    def is_ready(self):
        return self.model is not None and self.trained and self.is_fitted

class LSTMModelManager:
    def __init__(self, model_path: str = 'lstm-model', input_shape=(60, 17), units=50, lr=0.001):
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.units = units
        self.lr = lr
        self._cache: Dict[str, LSTMModel] = {}
        self._lock = threading.Lock()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.logger = logger

    async def initialize_models(self):
        self.logger.info("Initializing and loading all LSTM models...")
        loop = asyncio.get_running_loop()
        executor = self._get_executor()
        
        tasks = []
        for model_file in self.model_path.glob("model_*.keras"):
            task = loop.run_in_executor(executor, self._load_single_model, model_file)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        self.logger.info(f"Finished loading models. Total cached models: {len(self._cache)}")

    def _load_single_model(self, model_file: Path):
        try:
            filename = model_file.stem
            parts = filename.split('_')
            if len(parts) > 1:
                symbol_tf = parts[1]
                symbol_parts = symbol_tf.split('-')
                symbol = symbol_parts[0].upper()
                timeframe = '-'.join(symbol_parts[1:])
                
                key = f"{symbol}-{timeframe}"
                with self._lock:
                    if key in self._cache:
                        return
                
                self.logger.debug(f"Loading model for {key} from {model_file}...")
                model = LSTMModel(symbol=symbol, timeframe=timeframe, model_path=self.model_path,
                                input_shape=self.input_shape, units=self.units, lr=self.lr)
                if model.is_ready():
                    with self._lock:
                        self._cache[key] = model
                else:
                    self.logger.warning(f"Model for {key} loaded but is not ready. It may need training.")
        except Exception as e:
            self.logger.error(f"Failed to parse and load model from file {model_file}: {e}")

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self.executor is None or getattr(self.executor, '_shutdown', True):
                self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
            return self.executor

    async def _run_in_executor(self, func, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            executor = self._get_executor()
            return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
        except Exception as e:
            self.logger.error(f"Error running function in executor: {e}")
            return None

    def get_model(self, symbol: str, timeframe: str) -> Optional[LSTMModel]:
        key = f"{symbol}-{timeframe}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        
        model = LSTMModel(symbol=symbol, timeframe=timeframe, model_path=self.model_path,
                        input_shape=self.input_shape, units=self.units, lr=self.lr)
        
        with self._lock:
            self._cache[key] = model
        return model

    async def train_model_if_needed(self, symbol: str, timeframe: str, data: pd.DataFrame):
        model = self.get_model(symbol, timeframe)
        if model and not model.is_ready():
            self.logger.info(f"Model for {symbol}-{timeframe} is not trained. Training now...")
            success = await self._run_in_executor(model.fit, data)
            if success:
                self.logger.info(f"Successfully trained model for {symbol}-{timeframe}.")
            else:
                self.logger.error(f"Failed to train model for {symbol}-{timeframe}.")

    async def predict_async(self, symbol: str, timeframe: str, data: pd.DataFrame):
        model = self.get_model(symbol, timeframe)
        if model and model.is_ready():
            return await self._run_in_executor(model.predict, data)
        return None

    def cleanup(self):
        self.logger.info("Cleaning up LSTMModelManager...")
        with self._lock:
            for model in self._cache.values():
                model.clear_cache()
            self._cache.clear()
            if self.executor and not getattr(self.executor, '_shutdown', True):
                self.executor.shutdown(wait=True, cancel_futures=False)
                self.executor = None
        self.logger.info("LSTMModelManager cleanup complete.")