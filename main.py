import asyncio
import signal
import sys
from typing import List

from services.trading_service import TradingService
from data.data_provider import MarketDataProvider
from config.settings import ConfigManager
from config.logger import logger
from common.core import TradingSignal


class MainApp:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.trading_service = None

    def _handle_shutdown(self, sig, frame):
        logger.info(f"Shutdown signal {sig} received. Initiating graceful shutdown...")
        self.shutdown_event.set()

    async def _run_analysis_cycle(self):
        if not self.trading_service:
            logger.error("TradingService not initialized.")
            return

        try:
            signals: List[TradingSignal] = await self.trading_service.run_analysis_for_all_symbols()
            if signals:
                logger.info("Top signals for this cycle:")
                for s in signals:
                    logger.info(
                        f"  - {s.symbol} ({s.timeframe}): {s.signal_type.value} | "
                        f"Entry: {s.entry_price:.4f}, Exit: {s.exit_price:.4f}, SL: {s.stop_loss:.4f} | "
                        f"Confidence: {s.confidence_score:.2f}, R/R: {s.risk_reward_ratio:.2f}"
                    )
            else:
                logger.info("No actionable signals found in this cycle.")

        except Exception as e:
            logger.error(f"An error occurred during the analysis cycle: {e}", exc_info=True)

    async def run(self):
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Initializing application components...")
        config_manager = ConfigManager()
        market_data_provider = MarketDataProvider(config_manager=config_manager)
        self.trading_service = TradingService(
            market_data_provider=market_data_provider,
            config_manager=config_manager,
        )

        analysis_interval_hours = config_manager.get("analysis_interval_hours", 4)
        logger.info(f"Starting main analysis loop. Cycle interval: {analysis_interval_hours} hours.")

        while not self.shutdown_event.is_set():
            await self._run_analysis_cycle()

            try:
                # Wait for the next cycle or until shutdown is triggered
                await asyncio.wait_for(
                    self.shutdown_event.wait(),
                    timeout=analysis_interval_hours * 3600
                )
            except asyncio.TimeoutError:
                continue  # Timeout reached, continue to the next analysis cycle
            except Exception as e:
                logger.error(f"Error in main loop wait: {e}")
                break

        logger.info("Shutdown event set. Cleaning up resources...")
        if self.trading_service:
            await self.trading_service.cleanup()
        logger.info("Application has been shut down gracefully.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    app = MainApp()
    try:
        asyncio.run(app.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application interrupted by user. Shutting down.")