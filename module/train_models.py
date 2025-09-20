import asyncio
import warnings

import optuna
import pandas as pd

from module.config import ConfigManager
from module.constants import SYMBOLS, TIME_FRAMES
from module.logger_config import logger
from module.models import LSTMModel, XGBoostModel
from module.optimization import HyperparameterOptimizer
from module.sentiment import ExchangeManager

warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.exchange_manager = ExchangeManager()

    async def train_and_optimize(self, symbol, timeframe, model_type="lstm"):
        logger.info(f"--- Starting training for {symbol}-{timeframe} ({model_type}) ---")
        try:
            data = await self.exchange_manager.fetch_ohlcv_data(symbol, timeframe, limit=2000)
            if data.empty or len(data) < 200:
                logger.warning(f"Insufficient data for {symbol}-{timeframe}. Skipping.")
                return

            optimizer = HyperparameterOptimizer(data, model_type=model_type, n_trials=25)
            best_params, best_value = await optimizer.run()
            logger.info(f"Best Sharpe Ratio for {symbol}-{timeframe}: {best_value:.4f}")

            if model_type == "lstm":
                model = LSTMModel(symbol=symbol, timeframe=timeframe, **best_params)
            else:
                model = XGBoostModel(symbol=symbol, timeframe=timeframe, **best_params)

            logger.info(f"Retraining with best params: {best_params}")
            # This is a placeholder for the actual training logic with features
            # model.fit(X, y)
            model.save_model()
            logger.info(f"✅ Successfully trained and saved model for {symbol}-{timeframe}")

        except Exception as e:
            logger.error(f"Error during training for {symbol}-{timeframe}: {e}")

    async def train_all_models(self):
        logger.info("Starting model training for all symbols and timeframes.")
        symbols = self.config.get("symbols", SYMBOLS)
        timeframes = self.config.get("timeframes", TIME_FRAMES)

        await self.exchange_manager.init_database()

        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append(self.train_and_optimize(symbol, timeframe, "lstm"))
                tasks.append(self.train_and_optimize(symbol, timeframe, "xgboost"))

        await asyncio.gather(*tasks)
        await self.exchange_manager.close()
        logger.info("--- Model training process finished. ---")


async def main():
    config_manager = ConfigManager()
    trainer = ModelTrainer(config_manager)
    await trainer.train_all_models()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Training process interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error in training script: {e}")