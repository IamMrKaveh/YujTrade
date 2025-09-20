import asyncio
import warnings
from pathlib import Path

import pandas as pd

from module.config import ConfigManager
from module.constants import SYMBOLS, TIME_FRAMES
from module.logger_config import logger
from module.lstm import LSTMModel
from module.sentiment import ExchangeManager

warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.exchange_manager = ExchangeManager()

    async def train_all_models(self):
        """
        Iterates through all symbols and timeframes, fetches data,
        and trains an LSTM model for each combination.
        """
        logger.info("Starting LSTM model training for all symbols and timeframes.")
        
        symbols = self.config.get('symbols', SYMBOLS)
        timeframes = self.config.get('timeframes', TIME_FRAMES)
        
        await self.exchange_manager.init_database()
        
        total_models = len(symbols) * len(timeframes)
        trained_count = 0
        failed_count = 0
        
        for i, symbol in enumerate(symbols):
            for j, timeframe in enumerate(timeframes):
                model_identifier = f"{symbol}-{timeframe}"
                logger.info(f"--- Training model {i*len(timeframes) + j + 1}/{total_models}: {model_identifier} ---")
                
                try:
                    # Fetch a large dataset for training
                    data = await self.exchange_manager.fetch_ohlcv_data(symbol, timeframe, limit=2000)
                    
                    if data.empty or len(data) < 200:
                        logger.warning(f"Insufficient data for {model_identifier} ({len(data)} candles). Skipping.")
                        continue
                    
                    # Initialize a specific model for the symbol and timeframe
                    model = LSTMModel(symbol=symbol, timeframe=timeframe)
                    
                    # The fit method now handles sequence preparation and training
                    success = await asyncio.to_thread(
                        model.fit,
                        data,
                        epochs=50,  # More epochs for offline training
                        batch_size=32,
                        verbose=1,
                        validation_split=0.15
                    )
                    
                    if success:
                        logger.info(f"✅ Successfully trained and saved model for {model_identifier}")
                        trained_count += 1
                    else:
                        logger.error(f"❌ Failed to train model for {model_identifier}")
                        failed_count += 1
                        
                    # Clean up to release memory
                    model.clear_cache()
                    del model
                    
                except Exception as e:
                    logger.error(f"An unexpected error occurred during training for {model_identifier}: {e}")
                    failed_count += 1
                
                # Small delay to avoid overwhelming systems
                await asyncio.sleep(2)
                
        await self.exchange_manager.close()
        logger.info("--- Training Summary ---")
        logger.info(f"Total models attempted: {total_models}")
        logger.info(f"✅ Successfully trained: {trained_count}")
        logger.info(f"❌ Failed to train: {failed_count}")
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