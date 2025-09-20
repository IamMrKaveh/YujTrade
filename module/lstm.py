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

from module.logger_config import logger

tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')

class LSTMModel:
    def __init__(self, input_shape=(60, 15), units=64, lr=0.001, model_path='lstm-model', symbol=None, timeframe=None):
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
        
    def __del__(self):
        self.clear_cache()

    def clear_cache(self):
        try:
            if hasattr(self, 'model') and self.model:
                del self.model
                self.model = None
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error during model cleanup: {e}")

    def _setup_gpu(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except (RuntimeError, AttributeError):
                    pass
        except Exception:
            pass

    def _get_model_paths(self):
        if self.symbol and self.timeframe:
            model_suffix = f"_{self.symbol.lower().replace('/', '')}-{self.timeframe}"
        elif self.symbol:
            model_suffix = f"_{self.symbol.lower().replace('/', '')}"
        else:
            model_suffix = ""
            
        model_file = self.model_path / f"model{model_suffix}.keras"
        scaler_file = self.model_path / f"scaler{model_suffix}.pkl"
        return model_file, scaler_file

    def _load_or_create_model(self, units, lr):
        try:
            model_file, scaler_file = self._get_model_paths()
            
            if model_file.exists() and scaler_file.exists():
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        self.model = tf.keras.models.load_model(str(model_file), compile=False)
                        if self.model:
                            self.model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), 
                                            loss='huber', metrics=['mae'])
                    
                    with open(scaler_file, 'rb') as f:
                        self.scaler = pickle.load(f)
                    
                    self.trained = True
                    self.is_fitted = True
                    if self.logger:
                        self.logger.info(f"Loaded existing model and scaler from {model_file}")
                    return
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to load model {model_file}: {e}. Creating a new one.")
                    if self.model:
                        try:
                            del self.model
                        except:
                            pass
                        self.model = None
            
            self._create_model(units, lr)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in _load_or_create_model: {e}")
            self.model = None

    def _create_model(self, units, lr):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
                if self.logger:
                    self.logger.info("Created a new LSTM model instance.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create LSTM model: {e}")
            self.model = None

    def save_model(self):
        try:
            if not self.model or not hasattr(self, 'scaler'):
                return False
            
            model_file, scaler_file = self._get_model_paths()
            
            if self.model and self.trained:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    self.model.save(str(model_file))
            
            if self.is_fitted and hasattr(self.scaler, 'scale_'):
                import joblib
                joblib.dump(self.scaler, str(scaler_file))
            
            if self.logger:
                self.logger.info(f"Saved model to {model_file} and scaler to {scaler_file}")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {e}")
            return False

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < 50:
            return pd.DataFrame()
        
        df = data.copy()
        features_df = pd.DataFrame(index=df.index)
        
        features_df['close_norm'] = (df['close'].rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False))
        features_df['volume_norm'] = (df['volume'].rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False))
        features_df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features_df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        try:
            import pandas_ta as ta
            features_df['rsi'] = ta.rsi(df['close'], length=14).fillna(50) / 100.0
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                macd_line = macd.iloc[:, 0].fillna(0)
                features_df['macd'] = np.tanh(macd_line / df['close'])
            else:
                features_df['macd'] = 0.0

            bb = ta.bbands(df['close'], length=20)
            if bb is not None and not bb.empty:
                features_df['bb_position'] = ((df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])).fillna(0.5)
            else:
                features_df['bb_position'] = 0.5
            
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            if stoch is not None and not stoch.empty:
                features_df['stoch'] = stoch.iloc[:, 0].fillna(50) / 100.0
            else:
                features_df['stoch'] = 0.5

            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            features_df['atr'] = (atr / df['close']).fillna(0.02)
        except Exception:
            features_df['rsi'] = 0.5
            features_df['macd'] = 0
            features_df['bb_position'] = 0.5
            features_df['stoch'] = 0.5
            features_df['atr'] = 0.02
        
        features_df['sma_5'] = df['close'].rolling(5).mean() / df['close']
        features_df['sma_20'] = df['close'].rolling(20).mean() / df['close']
        features_df['ema_12'] = df['close'].ewm(span=12).mean() / df['close']
        features_df['ema_26'] = df['close'].ewm(span=26).mean() / df['close']
        
        features_df['price_change_1'] = df['close'].pct_change(1)
        features_df['price_change_5'] = df['close'].pct_change(5)
        features_df['volume_change'] = df['volume'].pct_change(1)
        
        if hasattr(df.index, 'hour') and hasattr(df.index, 'dayofweek'):
            features_df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features_df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        else:
            features_df['hour_sin'] = 0
            features_df['day_sin'] = 0
        
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        for col in features_df.columns:
            features_df[col] = features_df[col].replace([np.inf, -np.inf], 0)
            col_values = features_df[col].values
            if len(col_values) > 20:
                rolling_mean = pd.Series(col_values).rolling(20, min_periods=1).mean()
                rolling_std = pd.Series(col_values).rolling(20, min_periods=1).std()
                features_df[col] = (col_values - rolling_mean) / (rolling_std + 1e-8)
            features_df[col] = np.clip(features_df[col], -3, 3)
        
        return features_df.iloc[:, :self.input_shape[1]]
    
    def _validate_dataframe(self, data):
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True

    def prepare_sequences(self, data, window=None, for_training=True):
        try:
            if window is None:
                window = self.input_shape[0]
            
            if isinstance(data, pd.DataFrame):
                self._validate_dataframe(data)
                features_df = self._create_features(data)
                if features_df.empty:
                    return np.array([]), np.array([])
                
                feature_values = features_df.values
                target_values = data['close'].values
            else:
                if isinstance(data, pd.Series):
                    values = data.values
                else:
                    values = np.array(data)
                
                if len(values) == 0:
                    return np.array([]), np.array([])
                
                feature_values = values.reshape(-1, 1)
                if self.input_shape[1] > 1:
                    padding = np.zeros((feature_values.shape[0], self.input_shape[1] - 1))
                    feature_values = np.hstack([feature_values, padding])
                target_values = values
            
            if len(feature_values) < window + 10:
                return np.array([]), np.array([])
            
            target_values = pd.to_numeric(target_values, errors='coerce')
            nan_mask = np.isnan(target_values) | np.isinf(target_values)
            
            if np.all(nan_mask):
                return np.array([]), np.array([])
            
            if np.any(nan_mask):
                for i in range(len(target_values)):
                    if nan_mask[i]:
                        if i > 0:
                            target_values[i] = target_values[i-1]
                        else:
                            target_values[i] = np.nanmean(target_values[~nan_mask])
            
            valid_mask = ~(np.isnan(target_values) | np.isinf(target_values))
            feature_values = feature_values[valid_mask]
            target_values = target_values[valid_mask]
            
            if len(target_values) < window + 10:
                return np.array([]), np.array([])
            
            if np.all(target_values == target_values[0]) or np.std(target_values) == 0:
                if for_training:
                    return np.array([]), np.array([])
                else:
                    return None
            
            if for_training:
                try:
                    clean_target = target_values[~np.isnan(target_values)]
                    if len(clean_target) == 0:
                        return np.array([]), np.array([])
                    self.scaler.fit(clean_target.reshape(-1, 1))
                    self.is_fitted = True
                    target_scaled = self.scaler.transform(target_values.reshape(-1, 1)).flatten()
                    target_scaled = np.nan_to_num(target_scaled, nan=0.5)
                except Exception:
                    return np.array([]), np.array([])
            else:
                if not self.is_fitted or not hasattr(self.scaler, 'scale_'):
                    return None
                try:
                    target_scaled = self.scaler.transform(target_values.reshape(-1, 1)).flatten()
                    target_scaled = np.nan_to_num(target_scaled, nan=0.5)
                except Exception:
                    return None
            
            X, y = [], []
            for i in range(window, len(feature_values)):
                try:
                    X.append(feature_values[i-window:i])
                    if i < len(target_scaled):
                        y.append(target_scaled[i])
                except IndexError:
                    break
            
            if len(X) == 0 or len(y) == 0:
                return np.array([]), np.array([])
            
            X = np.array(X)
            y = np.array(y)
            
            if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
                y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return X, y
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in prepare_sequences: {e}")
            return np.array([]), np.array([])

    def fit(self, data, epochs=15, batch_size=32, verbose=0, validation_split=0.2):
        try:
            if self.trained or not self.model:
                return self.trained
            
            X, y = self.prepare_sequences(data, for_training=True)
            
            if X.size == 0 or y.size == 0:
                return False
            
            if X.shape[0] != y.shape[0]:
                min_len = min(X.shape[0], y.shape[0])
                X = X[:min_len]
                y = y[:min_len]
            
            epochs = max(10, min(epochs, 100))
            batch_size = max(8, min(batch_size, len(X) // 4)) if len(X) >= 32 else max(2, len(X) // 8)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-7, verbose=0)
            ]
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                history = self.model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=verbose,
                    shuffle=False
                )
            
            if history and 'loss' in history.history and len(history.history['loss']) > 0:
                final_loss = history.history['loss'][-1]
                val_loss = history.history.get('val_loss', [float('inf')])[-1]
                if (final_loss < float('inf') and not np.isnan(final_loss) and 
                    val_loss < float('inf') and not np.isnan(val_loss) and val_loss < 10):
                    self.trained = True
                    self.save_model()
                    return True
            
            return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in fit: {e}")
            self.trained = False
            return False

    def predict(self, data):
        try:
            if not self.model or not self.trained or not self.is_fitted:
                return None
            
            if isinstance(data, pd.DataFrame):
                self._validate_dataframe(data)
                features_df = self._create_features(data)
                if features_df.empty:
                    return None
                
                last_sequence = features_df.tail(self.input_shape[0]).values
            else:
                if isinstance(data, pd.Series):
                    values = data.values
                else:
                    values = np.array(data)
                
                if len(values) < self.input_shape[0]:
                    return None
                
                last_sequence = values[-self.input_shape[0]:].reshape(-1, 1)
                if self.input_shape[1] > 1:
                    padding = np.zeros((last_sequence.shape[0], self.input_shape[1] - 1))
                    last_sequence = np.hstack([last_sequence, padding])
            
            if np.any(np.isnan(last_sequence)) or np.any(np.isinf(last_sequence)):
                last_sequence = np.nan_to_num(last_sequence, nan=0.0, posinf=1.0, neginf=-1.0)
            
            X = last_sequence.reshape(1, self.input_shape[0], self.input_shape[1])
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                prediction = self.model.predict(X, verbose=0)
            
            if prediction is None or len(prediction) == 0:
                return None
            
            if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
                if np.any(self.scaler.scale_ != 0):
                    try:
                        prediction_scaled = self.scaler.inverse_transform(prediction.reshape(-1, 1))
                        return prediction_scaled.flatten()
                    except Exception:
                        return None
                else:
                    return None
            else:
                return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in predict: {e}")
            return None

    def is_ready(self):
        return (self.model is not None and 
                self.trained and 
                self.is_fitted and 
                hasattr(self.scaler, 'scale_'))

class LSTMModelManager:
    def __init__(self, model_path: str = 'lstm-model', input_shape=(60, 15), units=50, lr=0.001):
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.units = units
        self.lr = lr
        self._cache: Dict[str, LSTMModel] = {}
        self._lock = threading.Lock()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.logger = logger

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self.executor is None or getattr(self.executor, '_shutdown', True):
                self.executor = ThreadPoolExecutor(max_workers=2)
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
        
        try:
            self.logger.info(f"Loading/creating model for {key}...")
            model = LSTMModel(
                symbol=symbol,
                timeframe=timeframe,
                model_path=self.model_path,
                input_shape=self.input_shape,
                units=self.units,
                lr=self.lr
            )
            if model.is_ready():
                with self._lock:
                    self._cache[key] = model
                return model
            else:
                self.logger.warning(f"Model for {key} is not ready (likely not trained).")
                return None
        except Exception as e:
            self.logger.error(f"Failed to get model for {key}: {e}")
            return None

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
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.executor = None
        self.logger.info("LSTMModelManager cleanup complete.")