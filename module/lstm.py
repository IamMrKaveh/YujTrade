import asyncio
import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, BatchNormalization, Dense, Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas_ta as ta

from module.logger_config import logger

tf.get_logger().setLevel('ERROR')
# Attempt to disable GPU usage if not needed or causing issues
try:
    tf.config.set_visible_devices([], 'GPU')
except (RuntimeError, ValueError) as e:
    logger.warning(f"Could not disable GPU visibility. This might be fine if you intend to use a GPU. Error: {e}")


class LSTMModel:
    def __init__(self, model_path: Path, symbol: str, timeframe: str, input_shape=(60, 17), units=64, lr=0.001):
        self.model: Optional[Sequential] = None
        self.input_shape = input_shape
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = model_path
        self.is_trained = False
        self._create_model(units, lr)

    def _create_model(self, units, lr):
        try:
            self.model = Sequential([
                LSTM(units, input_shape=self.input_shape, return_sequences=True, dropout=0.15, recurrent_dropout=0.15),
                LSTM(units // 2, return_sequences=True, dropout=0.15, recurrent_dropout=0.15),
                LSTM(units // 4, return_sequences=False, dropout=0.1, recurrent_dropout=0.1),
                Dropout(0.25),
                Dense(50, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='linear')
            ])
            optimizer = Adam(learning_rate=lr, clipnorm=1.0)
            self.model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
            logger.debug(f"Created a new LSTM model instance for {self.symbol}-{self.timeframe}.")
        except Exception as e:
            logger.error(f"Failed to create LSTM model for {self.symbol}-{self.timeframe}: {e}")
            self.model = None

    def fit(self, X, y, epochs=15, batch_size=32, validation_split=0.2):
        if not self.model:
            return False
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-7)
        ]
        
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks,
                                verbose=0, shuffle=False)
        
        final_val_loss = history.history.get('val_loss', [float('inf')])[-1]
        if final_val_loss < 1.0: # Arbitrary threshold for successful training
            self.is_trained = True
            return True
        return False

    def predict(self, X):
        if not self.model or not self.is_trained:
            return None
        return self.model.predict(X, verbose=0)
        
    def clear_session(self):
        if self.model:
            del self.model
            self.model = None
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            logger.debug(f"Cleared TF session for model {self.symbol}-{self.timeframe}")


class LSTMModelManager:
    def __init__(self, model_path: str = 'MLM', input_shape=(60, 17), units=50, lr=0.001):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.input_shape = input_shape
        self.units = units
        self.lr = lr
        self._cache: Dict[str, Tuple[LSTMModel, MinMaxScaler]] = {}
        self._lock = threading.Lock()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.logger = logger

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self.executor is None or getattr(self.executor, '_shutdown', True):
                self.executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1))
            return self.executor

    async def _run_in_executor(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        executor = self._get_executor()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    def _get_model_paths(self, symbol: str, timeframe: str) -> Tuple[Path, Path]:
        model_suffix = f"_{symbol.lower().replace('/', '')}-{timeframe}"
        model_file = self.model_path / f"model{model_suffix}.keras"
        scaler_file = self.model_path / f"scaler{model_suffix}.pkl"
        return model_file, scaler_file

    def get_model(self, symbol: str, timeframe: str) -> Optional[Tuple[LSTMModel, MinMaxScaler]]:
        key = f"{symbol}-{timeframe}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        
        model_file, scaler_file = self._get_model_paths(symbol, timeframe)
        
        model = LSTMModel(model_path=self.model_path, symbol=symbol, timeframe=timeframe, 
                          input_shape=self.input_shape, units=self.units, lr=self.lr)
        scaler = MinMaxScaler()
        
        if model_file.exists() and scaler_file.exists():
            try:
                model.model = tf.keras.models.load_model(str(model_file), compile=False)
                model.model.compile(optimizer=Adam(learning_rate=self.lr, clipnorm=1.0), loss='huber', metrics=['mae'])
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                model.is_trained = True
                self.logger.info(f"Loaded existing model and scaler for {key} from {model_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load model {model_file} for {key}: {e}. A new one will be used.")
        
        with self._lock:
            self._cache[key] = (model, scaler)
        return model, scaler

    async def initialize_models(self):
        self.logger.info("Initializing and loading all LSTM models...")
        tasks = [
            self._run_in_executor(self.get_model, file.stem.split('_')[1].split('-')[0].upper(), '-'.join(file.stem.split('_')[1].split('-')[1:]))
            for file in self.model_path.glob("model_*.keras")
        ]
        await asyncio.gather(*tasks)
        self.logger.info(f"Finished loading models. Total cached models: {len(self._cache)}")

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        try:
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.sma(length=20, append=True); df.ta.sma(length=50, append=True)
            df.ta.ema(length=12, append=True); df.ta.ema(length=26, append=True)
            
            feature_cols = [col for col in df.columns if col.startswith(('RSI', 'MACD', 'BBL', 'BBM', 'BBU', 'STOCH', 'ATR', 'SMA', 'EMA'))]
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
        
        features_df = features_df.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
        
        current_num_features = features_df.shape[1]
        target_num_features = self.input_shape[1]
        if current_num_features > target_num_features:
            features_df = features_df.iloc[:, :target_num_features]
        elif current_num_features < target_num_features:
            padding = pd.DataFrame(0, index=features_df.index, columns=[f'pad_{i}' for i in range(target_num_features - current_num_features)])
            features_df = pd.concat([features_df, padding], axis=1)

        return features_df

    def _prepare_sequences(self, data: pd.DataFrame, scaler: MinMaxScaler, for_training: bool) -> Tuple[np.ndarray, np.ndarray]:
        window = self.input_shape[0]
        if len(data) < window: return np.array([]), np.array([])

        features_df = self._create_features(data)
        if features_df.empty: return np.array([]), np.array([])
        
        feature_values = features_df.values
        target_values = data['close'].values

        if for_training:
            scaler.fit(target_values.reshape(-1, 1))
        
        target_scaled = scaler.transform(target_values.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(window, len(feature_values)):
            X.append(feature_values[i-window:i])
            y.append(target_scaled[i])
        
        if not X: return np.array([]), np.array([])

        X, y = np.array(X), np.array(y)
        return np.nan_to_num(X), np.nan_to_num(y)

    async def train_model_if_needed(self, symbol: str, timeframe: str, data: pd.DataFrame):
        model_tuple = self.get_model(symbol, timeframe)
        if not model_tuple: return

        model, scaler = model_tuple
        if not model.is_trained:
            self.logger.info(f"Model for {symbol}-{timeframe} is not trained. Training now...")
            
            def training_job():
                X, y = self._prepare_sequences(data, scaler, for_training=True)
                if X.size == 0 or y.size == 0: return False
                success = model.fit(X, y)
                if success:
                    model_file, scaler_file = self._get_model_paths(symbol, timeframe)
                    model.model.save(str(model_file))
                    with open(scaler_file, 'wb') as f: pickle.dump(scaler, f)
                return success

            success = await self._run_in_executor(training_job)
            if success:
                self.logger.info(f"Successfully trained model for {symbol}-{timeframe}.")
            else:
                self.logger.error(f"Failed to train model for {symbol}-{timeframe}.")

    async def predict_async(self, symbol: str, timeframe: str, data: pd.DataFrame):
        model_tuple = self.get_model(symbol, timeframe)
        if not model_tuple: return None
        
        model, scaler = model_tuple
        if not model.is_trained: return None

        def prediction_job():
            window = self.input_shape[0]
            if len(data) < window: return None
            
            features_df = self._create_features(data)
            if features_df.empty: return None

            last_sequence = features_df.tail(window).values
            X = np.nan_to_num(last_sequence).reshape(1, window, self.input_shape[1])
            
            prediction_scaled = model.predict(X)
            if prediction_scaled is None: return None

            return scaler.inverse_transform(prediction_scaled).flatten()

        return await self._run_in_executor(prediction_job)

    def cleanup(self):
        self.logger.info("Cleaning up LSTMModelManager...")
        with self._lock:
            for key, (model, _) in self._cache.items():
                model.clear_session()
            self._cache.clear()
            if self.executor and not getattr(self.executor, '_shutdown', True):
                self.executor.shutdown(wait=True, cancel_futures=False)
                self.executor = None
        self.logger.info("LSTMModelManager cleanup complete.")