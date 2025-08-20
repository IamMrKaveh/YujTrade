import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import threading
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')

class LSTMModel:
    def __init__(self, input_shape=(60, 15), units=64, lr=0.001, model_path='lstm-model', symbol=None):
        self.model = None
        self.input_shape = input_shape
        self.trained = False
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.symbol = symbol
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.executor = None
        self._lock = threading.Lock()
        self.logger = None
        self._setup_gpu()
        self._load_or_create_model(units, lr)
        
    def __del__(self):
        self._cleanup_safely()

    def _cleanup_safely(self):
        try:
            with self._lock:
                if hasattr(self, 'executor') and self.executor and not getattr(self.executor, '_shutdown', True):
                    try:
                        self.executor.shutdown(wait=True, cancel_futures=True)
                    except:
                        pass
                    finally:
                        self.executor = None
                
                if hasattr(self, 'model') and self.model:
                    try:
                        del self.model
                        self.model = None
                        tf.keras.backend.clear_session()
                        import gc
                        gc.collect()
                    except:
                        pass
        except:
            pass

    def clear_cache(self):
        self._cleanup_safely()

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

    def _load_or_create_model(self, units, lr):
        try:
            symbol_suffix = f"_{self.symbol.lower()}" if self.symbol else ""
            model_file = self.model_path / f"model{symbol_suffix}.keras"
            scaler_file = self.model_path / f"scaler{symbol_suffix}.pkl"
            
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
                    return
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to load model: {e}")
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
        except Exception:
            self.model = None

    def _get_executor(self):
        with self._lock:
            if self.executor is None or getattr(self.executor, '_shutdown', True):
                self.executor = ThreadPoolExecutor(max_workers=2)
            return self.executor

    async def _run_in_executor(self, func, *args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            executor = self._get_executor()
            return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
        except Exception:
            return None

    def save_model(self):
        try:
            if not self.model or not hasattr(self, 'scaler'):
                return False
                
            symbol_suffix = f"_{self.symbol.lower()}" if self.symbol else ""
            model_file = self.model_path / f"model{symbol_suffix}.keras"
            scaler_file = self.model_path / f"scaler{symbol_suffix}.pkl"
            
            if self.model and self.trained:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    self.model.save(str(model_file))
            
            if self.is_fitted and hasattr(self.scaler, 'scale_'):
                import joblib
                joblib.dump(self.scaler, str(scaler_file))
            
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {e}")
            return False

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < 50:
            return pd.DataFrame()
        
        features_df = pd.DataFrame()
        
        features_df['close_norm'] = (data['close'].rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0))
        features_df['volume_norm'] = (data['volume'].rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0))
        features_df['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        features_df['close_open_ratio'] = (data['close'] - data['open']) / data['open']
        
        try:
            import pandas_ta as ta
            features_df['rsi'] = ta.rsi(data['close'], length=14).fillna(50) / 100.0
            macd_line = ta.macd(data['close'])['MACD_12_26_9'].fillna(0)
            features_df['macd'] = np.tanh(macd_line / data['close'])
            bb = ta.bbands(data['close'], length=20)
            features_df['bb_position'] = ((data['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])).fillna(0.5)
            features_df['stoch'] = ta.stoch(data['high'], data['low'], data['close'])['STOCHk_14_3_3'].fillna(50) / 100.0
            atr = ta.atr(data['high'], data['low'], data['close'], length=14)
            features_df['atr'] = (atr / data['close']).fillna(0.02)
        except:
            features_df['rsi'] = 0.5
            features_df['macd'] = 0
            features_df['bb_position'] = 0.5
            features_df['stoch'] = 0.5
            features_df['atr'] = 0.02
        
        features_df['sma_5'] = data['close'].rolling(5).mean() / data['close']
        features_df['sma_20'] = data['close'].rolling(20).mean() / data['close']
        features_df['ema_12'] = data['close'].ewm(span=12).mean() / data['close']
        features_df['ema_26'] = data['close'].ewm(span=26).mean() / data['close']
        
        features_df['price_change_1'] = data['close'].pct_change(1)
        features_df['price_change_5'] = data['close'].pct_change(5)
        features_df['volume_change'] = data['volume'].pct_change(1)
        
        if hasattr(data.index, 'hour') and hasattr(data.index, 'dayofweek'):
            features_df['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            features_df['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
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

    async def fit_async(self, data, epochs=15, batch_size=32, verbose=0, validation_split=0.2):
        try:
            return await self._run_in_executor(self.fit, data, epochs, batch_size, verbose, validation_split)
        except Exception:
            return False

    async def predict_async(self, data):
        try:
            return await self._run_in_executor(self.predict, data)
        except Exception:
            return None

    def is_ready(self):
        return (self.model is not None and 
                self.trained and 
                self.is_fitted and 
                hasattr(self.scaler, 'scale_'))
