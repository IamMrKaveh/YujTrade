import asyncio
import warnings
from typing import List

from module.config import ConfigManager
from module.constants import SYMBOLS, TIME_FRAMES
from module.logger_config import logger
from module.models import ModelManager, LSTMModel, XGBoostModel
from module.market import MarketDataProvider

warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, config: ConfigManager, market_data_provider: MarketDataProvider, model_manager: ModelManager):
        self.config = config
        self.market_data_provider = market_data_provider
        self.model_manager = model_manager

    async def train_and_save_model(self, symbol: str, timeframe: str, model_type: str):
        logger.info(f"--- Starting training for {symbol}-{timeframe} ({model_type}) ---")
        try:
            limit_map = {
                "1h": 2000,
                "4h": 1500,
                "1d": 1000,
                "1w": 500,
                "1M": 300
            }
            limit = limit_map.get(timeframe, 2000)
            
            data = await self.market_data_provider.fetch_ohlcv_data(symbol, timeframe, limit=limit)
            if data is None or data.empty or len(data) < 200:
                logger.warning(f"Insufficient data for {symbol}-{timeframe}. Skipping.")
                return

            model = await self.model_manager.get_model(model_type, symbol, timeframe)
            if not model:
                logger.error(f"Could not get model instance for {symbol}-{timeframe} ({model_type}).")
                return

            logger.info(f"Training {model_type} model for {symbol}-{timeframe}...")
            
            is_trained = await self.model_manager._run_in_executor(model.fit, data)

            if is_trained:
                await self.model_manager._run_in_executor(model.save)
                logger.info(f"✅ Successfully trained and saved {model_type} model for {symbol}-{timeframe}")
            else:
                logger.error(f"❌ Failed to train model for {symbol}-{timeframe} ({model_type}).")

        except Exception as e:
            logger.error(f"Error during training for {symbol}-{timeframe} ({model_type}): {e}", exc_info=True)

    async def train_all_models(self):
        logger.info("Starting model training for all symbols and timeframes.")
        symbols: List[str] = self.config.get("symbols", SYMBOLS)
        timeframes: List[str] = self.config.get("timeframes", TIME_FRAMES)
        model_types = ["lstm", "xgboost"]

        tasks = []
        if symbols and timeframes:
            for symbol in symbols:
                for timeframe in timeframes:
                    for model_type in model_types:
                        tasks.append(self.train_and_save_model(symbol, timeframe, model_type))

        await asyncio.gather(*tasks)
        logger.info("--- Model training process finished. ---")


async def main():
    config_manager = ConfigManager()
    market_data_provider = MarketDataProvider()
    model_manager = ModelManager(model_path='models')
    
    trainer = ModelTrainer(config_manager, market_data_provider, model_manager)
    await trainer.train_all_models()
    
    await market_data_provider.close()
    await model_manager.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Training process interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error in training script: {e}")
        
