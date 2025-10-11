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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.layers import (LSTM, BatchNormalization, Dense, Dropout) # type aignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import redis

from ..config.logger import logger
from .resource_manager import managed_tf_session
from ..common.exceptions import ModelError
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
            if data is None or data.empty or len(data) < self.sequence_length + 20: # Increased minimum data
                self.logger.warning(f"Insufficient data for {self.symbol}-{self.timeframe}, skipping training. Required: {self.sequence_length + 20}, Got: {len(data) if data is not None else 0}")
                return False
            
            features = self.feature_engineer.create_features(data)
            if features.empty or len(features) < self.sequence_length + 20:
                self.logger.warning(f"No features or insufficient features created for {self.symbol}-{self.timeframe}, skipping training.")
                return False
            
            scaled_features = self.feature_engineer.scale_features(features, fit=True)
            if scaled_features is None or len(scaled_features) < self.sequence_length + 20:
                self.logger.warning(f"Feature scaling failed or resulted in insufficient data for {self.symbol}-{self.timeframe}, skipping training.")
                return False

            self.input_shape = (self.sequence_length, scaled_features.shape[1])
            self._create_model()
            
            X, y = self.feature_engineer.create_sequences(scaled_features, scaled_features, self.sequence_length)
            if X.shape[0] == 0 or y.shape[0] == 0:
                self.logger.warning(f"Not enough data to create sequences for {self.symbol}-{self.timeframe}, skipping training.")
                return False

            # Ensure validation set is not empty
            if int(X.shape[0] * validation_split) < 1:
                self.logger.warning(f"Not enough data for validation split in {self.symbol}-{self.timeframe}. Disabling validation.")
                validation_split = 0.0

            effective_batch_size = min(batch_size or self.batch_size, X.shape[0] // 5)
            if effective_batch_size < 1:
                effective_batch_size = 1
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) if validation_split > 0 else EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-7) if validation_split > 0 else ReduceLROnPlateau(monitor='loss', factor=0.7, patience=3, min_lr=1e-7)
            ]
            
            with managed_tf_session():
                history = self.model.fit(X, y, epochs=epochs, batch_size=effective_batch_size,
                                        validation_split=validation_split if validation_split > 0 else 0.0, 
                                        callbacks=callbacks,
                                        verbose=0, shuffle=False)

            final_loss_key = 'val_loss' if validation_split > 0 else 'loss'
            final_val_loss = history.history.get(final_loss_key, [float('inf')])[-1]
            
            price_std = data['close'].std()
            loss_threshold = (0.25 * (price_std / data['close'].mean()) * 1.5) if price_std > 0 and data['close'].mean() > 0 else 0.75

            if final_val_loss < loss_threshold:
                self.is_trained = True
                self.last_training_date = pd.Timestamp.now(tz='UTC')
                try:
                    self.save()
                    self.logger.info(f"LSTM model for {self.symbol}-{self.timeframe} trained and saved successfully with {final_loss_key}: {final_val_loss:.4f}")
                except Exception as save_e:
                    self.logger.error(f"Failed to save LSTM model for {self.symbol}-{self.timeframe}: {save_e}")
                    self.is_trained = False
                    return False
                return True
            else:
                self.logger.warning(f"LSTM training for {self.symbol}-{self.timeframe} did not converge. Final {final_loss_key}: {final_val_loss:.4f} (Threshold: {loss_threshold:.4f})")
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
                # Make multiple predictions with dropout enabled for uncertainty estimation
                predictions_sample = [self.model(reshaped_sequence, training=True) for _ in range(10)]
                predictions_sample_np = np.array([p.numpy() for p in predictions_sample]).squeeze()


            if predictions_sample_np is None or len(predictions_sample_np) == 0:
                return None

            scaled_prediction = np.mean(predictions_sample_np, axis=0, keepdims=True)
            prediction_std = np.std(predictions_sample_np, axis=0)

            prediction = self.feature_engineer.inverse_scale_prediction(scaled_prediction)
            
            # Normalize uncertainty
            uncertainty = min(prediction_std * 0.5, 0.9)
            
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
            if self.model.input_shape:
                self.input_shape = self.model.input_shape[1:]
            else:
                self.logger.error(f"Loaded LSTM model for {self.symbol}-{self.timeframe} has no input_shape.")
                self.is_trained = False
                raise ModelError("Loaded model has no input_shape")

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
                self.logger.warning(f"No features or insufficient features created for XGBoost {self.symbol}-{self.timeframe}")
                return False
            
            target = features['close'].shift(-1)
            features = features.drop(columns=['close'])
            
            target = target.dropna()
            features = features.loc[target.index]
            
            if len(features) != len(target) or len(features) == 0:
                self.logger.warning(f"Feature-target mismatch for XGBoost {self.symbol}-{self.timeframe}")
                return False

            split_index = max(1, int(len(features) * (1 - validation_split)))
            X_train, X_val = features[:split_index], features[split_index:]
            y_train, y_val = target[:split_index], target[split_index:]
            
            if len(X_train) == 0 or len(X_val) == 0:
                self.logger.warning(f"Insufficient split data for XGBoost {self.symbol}-{self.timeframe}")
                return False

            self.model.fit(X_train.values, y_train.values, 
                          eval_set=[(X_val.values, y_val.values)], 
                          verbose=False,
                          early_stopping_rounds=10)
            
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
            
            last_features = features.drop(columns=['close']).tail(1)
            
            if last_features.empty:
                return None
            
            prediction = self.model.predict(last_features.values)
            
            uncertainty = 0.05
            if len(last_features) > 0:
                try:
                    # XGBoost doesn't have built-in dropout, so we simulate uncertainty
                    # by checking feature importance or other methods.
                    # For now, a fixed small uncertainty.
                    pass
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


