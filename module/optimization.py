import asyncio

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

from module.backtesting import BacktestingEngine
from module.logger_config import logger
from module.lstm import LSTMModel


class HyperparameterOptimizer:
    def __init__(self, data, model_type="lstm", n_trials=50):
        self.data = data
        self.model_type = model_type.lower()
        self.n_trials = n_trials

    def _define_lstm_search_space(self, trial):
        return {
            "units": trial.suggest_int("units", 32, 128, step=16),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }

    async def _objective(self, trial):
        try:
            if self.model_type == "lstm":
                params = self._define_lstm_search_space(trial)
                # Note: We are using the LSTMModel from lstm.py which is more integrated
                model = LSTMModel(**params)
            else:
                raise ValueError("Unsupported model type")

            # The BacktestingEngine needs a trading_service-like object
            # This part requires a more detailed implementation of how models are used in backtesting
            # For now, we assume a simplified path to get a performance metric
            # A full implementation would involve creating a mock trading_service
            # that uses the hyperparameter-tuned model.
            
            # Placeholder for a metric, as direct backtesting is complex here
            # In a real scenario, you would run a backtest and get the Sharpe ratio
            dummy_sharpe_ratio = trial.number * 0.1 # Example metric
            
            trial.report(dummy_sharpe_ratio, step=0)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return dummy_sharpe_ratio
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return -1.0

    async def run(self):
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        
        # Optuna's objective function needs to be synchronous if run with optimize
        # To use async, we'd need a more complex setup. Let's adapt.
        def sync_objective(trial):
            return asyncio.run(self._objective(trial))

        study.optimize(sync_objective, n_trials=self.n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logger.info("Study statistics: ")
        logger.info(f"  Number of finished trials: {len(study.trials)}")
        logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
        logger.info(f"  Number of complete trials: {len(complete_trials)}")

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info(f"  Value: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")

        return trial.params, trial.value