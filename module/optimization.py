import asyncio

from .backtesting import BacktestingEngine
from .logger_config import logger
from .models import LSTMModel


class HyperparameterOptimizer:
    def __init__(self, data, model_type="lstm", n_trials=50):
        self.data = data
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        # Check if optuna is available
        try:
            import optuna
            self.optuna = optuna
        except ImportError:
            self.optuna = None
            logger.error("Optuna is not installed. Please install it to use HyperparameterOptimizer: pip install optuna")

    def _define_lstm_search_space(self, trial):
        return {
            "units": trial.suggest_int("units", 32, 128, step=16),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }

    async def _objective(self, trial):
        if not self.optuna:
            raise ImportError("Optuna is not installed.")
        try:
            if self.model_type == "lstm":
                params = self._define_lstm_search_space(trial)
                model = LSTMModel(**params)
            else:
                raise ValueError("Unsupported model type")

            # This is a placeholder for a real performance metric.
            # A full implementation would run a backtest and return a value like the Sharpe ratio.
            dummy_sharpe_ratio = trial.number * 0.1
            
            trial.report(dummy_sharpe_ratio, step=0)

            if trial.should_prune():
                raise self.optuna.exceptions.TrialPruned()

            return dummy_sharpe_ratio
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return -1.0

    async def run(self):
        if not self.optuna:
            return {}, -1.0

        study = self.optuna.create_study(direction="maximize", pruner=self.optuna.pruners.MedianPruner())
        
        def sync_objective(trial):
            return asyncio.run(self._objective(trial))

        study.optimize(sync_objective, n_trials=self.n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[self.optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[self.optuna.trial.TrialState.COMPLETE])

        logger.info("Study statistics: ")
        logger.info(f"  Number of finished trials: {len(study.trials)}")
        logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
        logger.info(f"  Number of complete trials: {len(complete_trials)}")

        if not complete_trials:
            logger.error("No trials were completed successfully.")
            return {}, -1.0

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info(f"  Value: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")

        return trial.params, trial.value
    
