import asyncio
import pandas as pd
from typing import Optional

from backtesting.engine import BacktestingEngine
from config.logger import logger
from services.trading_service import TradingService
from data.data_provider import MarketDataProvider
from config.settings import ConfigManager
from utils.resource_manager import ResourceManager
from modeling.models import LSTMModel, XGBoostModel


class HyperparameterOptimizer:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        model_type="lstm",
        n_trials=50,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = data
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.loop = asyncio.get_running_loop()

        # Check if optuna is available
        try:
            import optuna

            self.optuna = optuna
        except ImportError:
            self.optuna = None
            logger.error(
                "Optuna is not installed. Please install it to use HyperparameterOptimizer: pip install optuna"
            )

    def _define_lstm_search_space(self, trial):
        return {
            "units": trial.suggest_int("units", 32, 256, step=32),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "sequence_length": trial.suggest_int("sequence_length", 20, 120, step=10),
        }

    def _define_xgboost_search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

    async def _objective(self, trial):
        if not self.optuna:
            raise ImportError("Optuna is not installed.")

        resource_manager: Optional[ResourceManager] = None
        trading_service: Optional[TradingService] = None

        try:
            # Setup a temporary trading service for backtesting
            config_manager = ConfigManager()
            resource_manager = ResourceManager()
            await resource_manager.get_session()
            await resource_manager.get_redis_client()

            market_data_provider = MarketDataProvider(
                resource_manager=resource_manager, config_manager=config_manager
            )
            await market_data_provider.initialize()

            trading_service = TradingService(
                market_data_provider=market_data_provider,
                config_manager=config_manager,
                resource_manager=resource_manager,
            )

            if self.model_type == "lstm":
                params = self._define_lstm_search_space(trial)
                model = LSTMModel(
                    symbol=self.symbol, timeframe=self.timeframe, **params
                )
            elif self.model_type == "xgboost":
                params = self._define_xgboost_search_space(trial)
                model = XGBoostModel(
                    symbol=self.symbol, timeframe=self.timeframe, **params
                )
            else:
                raise ValueError("Unsupported model type")

            # Train the model with the suggested parameters
            is_trained = await self.loop.run_in_executor(None, model.fit, self.data)
            if not is_trained:
                logger.warning(f"Trial {trial.number}: Model training failed. Pruning.")
                raise self.optuna.exceptions.TrialPruned()

            # Inject the trained model into the signal generator
            trading_service.signal_generator.model_manager._cache[
                f"{self.model_type}-{self.symbol}-{self.timeframe}"
            ] = model

            # Run backtest
            backtester = BacktestingEngine(trading_service)
            start_date = self.data.index.min().strftime("%Y-%m-%d")
            end_date = self.data.index.max().strftime("%Y-%m-%d")

            results = backtester.run_backtest(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start=start_date,
                end=end_date,
            )

            # Use Sharpe Ratio as the objective to maximize
            sharpe_ratio = results.get("sharpe_ratio", -1.0)

            if sharpe_ratio is None or pd.isna(sharpe_ratio):
                sharpe_ratio = -1.0

            trial.report(sharpe_ratio, step=0)

            if trial.should_prune():
                raise self.optuna.exceptions.TrialPruned()

            return sharpe_ratio
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            return -2.0  # Return a very low value for failed trials
        finally:
            # Cleanup resources
            if trading_service:
                await trading_service.cleanup()
            if resource_manager:
                await resource_manager.cleanup()

    async def run(self):
        if not self.optuna:
            return {}, -1.0

        study = self.optuna.create_study(
            direction="maximize", pruner=self.optuna.pruners.MedianPruner()
        )

        # Use run_in_executor to run the sync objective function in the event loop
        def sync_objective_wrapper(trial):
            future = asyncio.run_coroutine_threadsafe(self._objective(trial), self.loop)
            return future.result()

        # Run optimize in a separate thread to avoid blocking the event loop
        await self.loop.run_in_executor(
            None,
            lambda: study.optimize(
                sync_objective_wrapper, n_trials=self.n_trials, n_jobs=1
            ),
        )

        pruned_trials = study.get_trials(
            deepcopy=False, states=[self.optuna.trial.TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False, states=[self.optuna.trial.TrialState.COMPLETE]
        )

        logger.info("Study statistics: ")
        logger.info(f"  Number of finished trials: {len(study.trials)}")
        logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
        logger.info(f"  Number of complete trials: {len(complete_trials)}")

        if not complete_trials:
            logger.error("No trials were completed successfully.")
            return {}, -1.0

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info(f"  Value (Sharpe Ratio): {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")

        return trial.params, trial.value
