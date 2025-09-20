import asyncio

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

from module.backtesting import BacktestingEngine
from module.logger_config import logger
from module.models import LSTMModel, XGBoostModel


class HyperparameterOptimizer:
    def __init__(self, data, model_type="lstm", n_trials=50):
        self.data = data
        self.model_type = model_type.lower()
        self.n_trials = n_trials

    def _define_lstm_search_space(self, trial):
        return {
            "units": trial.suggest_int("units", 32, 128, step=16),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }

    def _define_xgboost_search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    async def _objective(self, trial):
        try:
            if self.model_type == "lstm":
                params = self._define_lstm_search_space(trial)
                model = LSTMModel(**params)
            elif self.model_type == "xgboost":
                params = self._define_xgboost_search_space(trial)
                model = XGBoostModel(**params)
            else:
                raise ValueError("Unsupported model type")

            backtest_engine = BacktestingEngine(self.data, model)
            result = await backtest_engine.run()

            sharpe_ratio = result.get("sharpe_ratio", 0)
            trial.report(sharpe_ratio, step=0)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return sharpe_ratio
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return -1.0

    async def run(self):
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: asyncio.run(self._objective(trial)), n_trials=self.n_trials)

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