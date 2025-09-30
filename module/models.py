import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Corrected imports for TensorFlow
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from module.logger_config import logger

tf.get_logger().setLevel("ERROR")


class BaseModel:
    def __init__(self, model_path: str, symbol: str, timeframe: str):
        self.model = None
        self.trained = False
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.logger = logger

    def _get_model_paths(self, extension: str):
        model_suffix = f"_{self.symbol.lower().replace('/', '')}-{self.timeframe}"
        model_file = self.model_path / f"model{model_suffix}.{extension}"
        scaler_file = self.model_path / f"scaler{model_suffix}.pkl"
        return model_file, scaler_file

    def save_model(self):
        raise NotImplementedError

    def predict(self, data: pd.DataFrame):
        raise NotImplementedError

    def is_ready(self) -> bool:
        return self.model is not None and self.trained

    def explain(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        self.logger.warning(f"Explanation not implemented for {self.__class__.__name__}")
        return None


class LSTMModel(BaseModel):
    def __init__(
        self,
        input_shape=(60, 15),
        units=64,
        lr=0.001,
        model_path="lstm-model",
        symbol=None,
        timeframe=None,
    ):
        super().__init__(model_path, symbol, timeframe)
        self.input_shape = input_shape
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self._load_or_create_model(units, lr)

    def _load_or_create_model(self, units, lr):
        model_file, scaler_file = self._get_model_paths("keras")
        if model_file.exists() and scaler_file.exists():
            try:
                self.model = tf.keras.models.load_model(str(model_file), compile=False)
                self.model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss="huber")
                with open(scaler_file, "rb") as f:
                    self.scaler = pickle.load(f)
                self.trained = self.is_fitted = True
                self.logger.info(f"Loaded LSTM model from {model_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load LSTM model {model_file}: {e}. Creating new.")
                self._create_model(units, lr)
        else:
            self._create_model(units, lr)

    def _create_model(self, units, lr):
        self.model = Sequential(
            [
                LSTM(units, input_shape=self.input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(units // 2, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        self.model.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
        self.logger.info("Created a new LSTM model.")

    def save_model(self):
        if not all([self.model, self.trained, self.is_fitted]):
            return False
        try:
            model_file, scaler_file = self._get_model_paths("keras")
            self.model.save(str(model_file))
            with open(scaler_file, "wb") as f:
                pickle.dump(self.scaler, f)
            return True
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {e}")
            return False

    def fit(self, X, y, epochs=20, batch_size=32, validation_split=0.1):
        if not self.model:
            return False
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
        )
        self.trained = True
        self.save_model()
        return True

    def predict(self, data):
        if not self.is_ready():
            return None
        try:
            X = data.reshape(1, self.input_shape[0], self.input_shape[1])
            prediction = self.model.predict(X, verbose=0)
            return self.scaler.inverse_transform(prediction).flatten()
        except Exception as e:
            self.logger.error(f"LSTM prediction error: {e}")
            return None

    def explain(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.is_ready():
            return None
        try:
            # SHAP DeepExplainer expects a background dataset. Using a sample of the data.
            background = data[np.random.choice(data.shape[0], 100, replace=False)]
            explainer = shap.DeepExplainer(self.model, background)
            shap_values = explainer.shap_values(data)
            return shap_values
        except Exception as e:
            self.logger.error(f"SHAP explanation for LSTM failed: {e}")
            return None


class XGBoostModel(BaseModel):
    def __init__(self, model_path="xgboost-model", symbol=None, timeframe=None, **params):
        super().__init__(model_path, symbol, timeframe)
        self.params = params or {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "tree_method": "hist", # Faster training
        }
        self._load_or_create_model()

    def _load_or_create_model(self):
        model_file, _ = self._get_model_paths("json")
        if model_file.exists():
            try:
                self.model = xgb.XGBRegressor(**self.params)
                self.model.load_model(model_file)
                self.trained = True
                self.logger.info(f"Loaded XGBoost model from {model_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load XGBoost model: {e}. Creating new.")
                self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
            self.logger.info("Created a new XGBoost model.")

    def save_model(self):
        if not self.model or not self.trained:
            return False
        try:
            model_file, _ = self._get_model_paths("json")
            self.model.save_model(model_file)
            return True
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {e}")
            return False

    def fit(self, X, y, validation_data=None):
        if not self.model:
            return False
        eval_set = [(X, y)]
        if validation_data:
            eval_set.append(validation_data)
        
        self.model.fit(X, y, eval_set=eval_set, early_stopping_rounds=10, verbose=False)
        self.trained = True
        self.save_model()
        return True

    def predict(self, data):
        if not self.is_ready():
            return None
        try:
            return self.model.predict(data)
        except Exception as e:
            self.logger.error(f"XGBoost prediction error: {e}")
            return None

    def explain(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.is_ready():
            return None
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(data)
            return shap_values
        except Exception as e:
            self.logger.error(f"SHAP explanation for XGBoost failed: {e}")
            return None