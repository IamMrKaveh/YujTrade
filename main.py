import asyncio
import platform
import signal
import sys
import traceback
import warnings

from telegram import Update

from module.config import Config, ConfigManager
from module.indicators import (ADXIndicator, ATRIndicator,
                               BollingerBandsIndicator, CCIIndicator,
                               ChaikinMoneyFlowIndicator, IchimokuIndicator,
                               MACDIndicator, MovingAverageIndicator,
                               OBVIndicator, RSIIndicator, StochasticIndicator,
                               SuperTrendIndicator, VolumeIndicator,
                               WilliamsRIndicator)
from module.logger_config import logger
from module.lstm import LSTMModelManager
from module.sentiment import ExchangeManager, OnChainFetcher, SentimentFetcher
from module.signals import (MultiTimeframeAnalyzer, SignalGenerator,
                              SignalRanking)
from module.telegram import TelegramBotHandler, TradingBotService

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')

_bot_instance = None

async def create_bot_application():
    global _bot_instance
    
    if _bot_instance is not None:
        logger.info("Bot instance already exists, cleaning up...")
        await _bot_instance.cleanup()
        _bot_instance = None
    
    config_manager = ConfigManager()
    logger.info("Configuration loaded successfully")
    
    application = None
    
    try:
        # 1. Centralized Service Initialization
        bot_token = Config.TELEGRAM_BOT_TOKEN
        if not bot_token:
            raise ValueError("Bot token is required")

        exchange_manager = ExchangeManager()
        
        cryptopanic_key = config_manager.get('cryptopanic_key', Config.CRYPTOPANIC_KEY)
        alchemy_url = config_manager.get('alchemy_url', Config.ALCHEMY_URL)

        sentiment_fetcher = SentimentFetcher(cryptopanic_key) if cryptopanic_key and cryptopanic_key.strip() else None
        if not sentiment_fetcher:
            logger.warning("Sentiment analysis disabled - no CryptoPanic API key")

        onchain_fetcher = OnChainFetcher(alchemy_url) if alchemy_url and alchemy_url.strip() else None
        if not onchain_fetcher:
            logger.warning("On-chain analysis disabled - no Alchemy URL")

        try:
            lstm_model_manager = LSTMModelManager(input_shape=(60, 15), units=50, lr=0.001)
            logger.info("LSTM Model Manager initialized")
        except Exception as e:
            logger.error(f"LSTM Model Manager initialization failed: {e}")
            lstm_model_manager = None

        indicators = {
            'sma_20': MovingAverageIndicator(20, "sma"), 'sma_50': MovingAverageIndicator(50, "sma"),
            'ema_12': MovingAverageIndicator(12, "ema"), 'ema_26': MovingAverageIndicator(26, "ema"),
            'rsi': RSIIndicator(), 'macd': MACDIndicator(), 'bb': BollingerBandsIndicator(),
            'stoch': StochasticIndicator(), 'volume': VolumeIndicator(), 'atr': ATRIndicator(),
            'ichimoku': IchimokuIndicator(), 'williams_r': WilliamsRIndicator(), 'cci': CCIIndicator(),
            'supertrend': SuperTrendIndicator(), 'adx': ADXIndicator(), 'cmf': ChaikinMoneyFlowIndicator(),
            'obv': OBVIndicator()
        }
        
        multi_tf_analyzer = MultiTimeframeAnalyzer(exchange_manager, indicators)
        
        signal_generator = SignalGenerator(
            sentiment_fetcher=sentiment_fetcher,
            onchain_fetcher=onchain_fetcher,
            lstm_model_manager=lstm_model_manager,
            multi_tf_analyzer=multi_tf_analyzer,
            config=config_manager.config
        )

        signal_ranking = SignalRanking()

        trading_service = TradingBotService(
            config=config_manager,
            exchange_manager=exchange_manager,
            signal_generator=signal_generator,
            signal_ranking=signal_ranking
        )

        # 2. Dependency Injection
        _bot_instance = TelegramBotHandler(
            bot_token=bot_token,
            config_manager=config_manager,
            trading_service=trading_service
        )
        
        await _bot_instance.initialize()
        
        application = _bot_instance.create_application()
        
        logger.info("Bot application created successfully")
        logger.info("Bot is ready and waiting for commands...")
        
        await application.initialize()
        await application.start()
        
        try:
            await application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
            stop_event = asyncio.Event()
            
            def signal_handler():
                logger.info("Received shutdown signal")
                stop_event.set()
            
            signal.signal(signal.SIGINT, lambda s, f: signal_handler())
            
            if platform.system() != 'Windows':
                signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
            
            await stop_event.wait()
                
        except (KeyboardInterrupt, SystemExit):
            logger.info("Received shutdown signal")
        finally:
            try:
                if application and hasattr(application, 'updater') and application.updater.running:
                    await application.updater.stop()
                    
                if application and application.running:
                    await application.stop()
                    
                if application:
                    await application.shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Bot crashed with error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
    finally:
        logger.info("Starting cleanup...")
        
        if _bot_instance:
            try:
                await _bot_instance.cleanup()
                logger.info("Bot handler cleanup completed")
            except Exception as e:
                logger.error(f"Error cleaning up bot handler: {e}")
            finally:
                _bot_instance = None
    
    logger.info("Bot shutdown completed successfully")

def main():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(create_bot_application())
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Main function error: {e}")
    finally:
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
                
            if loop and not loop.is_closed():
                pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                if pending_tasks:
                    logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
                    for task in pending_tasks:
                        task.cancel()
                        
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()