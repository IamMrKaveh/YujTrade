import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import numpy as np


class SignalTracker:
    def __init__(self, storage_path: str = "signal_history.json"):
        self.storage_path = Path(storage_path)
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_history(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def record(self, signal_id: str, outcome: bool, details: Dict[str, Any]):
        """
        Records the outcome of a trading signal.

        :param signal_id: A unique identifier for the signal.
        :param outcome: True for a successful trade (profit), False otherwise.
        :param details: A dictionary containing signal parameters for later analysis.
        """
        self.history[signal_id] = {"outcome": outcome, "details": details}
        self._save_history()

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Provides a summary of historical performance.
        """
        total_signals = len(self.history)
        if total_signals == 0:
            return {"total": 0, "win_rate": 0.0}

        wins = sum(1 for data in self.history.values() if data.get("outcome"))
        win_rate = (wins / total_signals) * 100
        return {"total": total_signals, "win_rate": win_rate}


class AdaptiveThresholdManager:
    def __init__(self):
        self.performance_history: Dict[str, List[Dict]] = {}
        self.min_samples = 50

    def record_performance(
        self,
        volatility_regime: str,
        hurst_range: str,
        threshold: float,
        signal_success: bool,
    ):
        key = f"{volatility_regime}_{hurst_range}"
        if key not in self.performance_history:
            self.performance_history[key] = []

        self.performance_history[key].append(
            {
                "threshold": threshold,
                "success": signal_success,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        if len(self.performance_history[key]) > 100:
            self.performance_history[key] = self.performance_history[key][-100:]

    def get_optimal_threshold(
        self, volatility_regime: str, hurst_range: str, default_threshold: float
    ) -> float:
        key = f"{volatility_regime}_{hurst_range}"

        if (
            key not in self.performance_history
            or len(self.performance_history[key]) < self.min_samples
        ):
            return default_threshold

        history = self.performance_history[key]
        threshold_performance = {}

        for record in history:
            threshold = round(record["threshold"], 2)
            if threshold not in threshold_performance:
                threshold_performance[threshold] = {"success": 0, "total": 0}

            threshold_performance[threshold]["total"] += 1
            if record["success"]:
                threshold_performance[threshold]["success"] += 1

        best_threshold = default_threshold
        best_accuracy = 0

        for threshold, perf in threshold_performance.items():
            if perf["total"] >= 10:
                accuracy = perf["success"] / perf["total"]
                confidence_interval = 1.96 * np.sqrt(
                    (accuracy * (1 - accuracy)) / perf["total"]
                )
                if accuracy - confidence_interval > 0.55:
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold

        return best_threshold


class MLConfidenceCalibrator:
    def __init__(self):
        self.calibration_data: Dict[str, List[Tuple[float, bool, float]]] = {}
        self.calibration_bins = 10

    def add_prediction(
        self,
        model_name: str,
        confidence: float,
        actual_result: bool,
        timestamp: datetime = None,
    ):
        if model_name not in self.calibration_data:
            self.calibration_data[model_name] = []

        time_decay_factor = 1.0
        if timestamp:
            days_old = (datetime.now(timezone.utc) - timestamp).days
            time_decay_factor = np.exp(-days_old / 90)

        self.calibration_data[model_name].append(
            (confidence, actual_result, time_decay_factor)
        )

        if len(self.calibration_data[model_name]) > 500:
            self.calibration_data[model_name] = self.calibration_data[model_name][-500:]

    def get_calibrated_confidence(
        self, model_name: str, raw_confidence: float
    ) -> float:
        """
        Returns a calibrated confidence score based on historical performance.
        """
        if (
            model_name not in self.calibration_data
            or len(self.calibration_data[model_name]) < 20
        ):
            return raw_confidence

        history = self.calibration_data[model_name]

        # Find the appropriate bin for the raw_confidence
        if raw_confidence >= 100:
            bin_index = self.calibration_bins - 1
        else:
            bin_index = int(raw_confidence * self.calibration_bins / 100)

        # Get all predictions in that bin
        bin_predictions = [
            (conf, result, weight)
            for conf, result, weight in history
            if int(conf * self.calibration_bins / 100) == bin_index
        ]

        if not bin_predictions:
            # If no data in this bin, check adjacent bins
            for offset in [-1, 1]:
                adjacent_bin_index = bin_index + offset
                if 0 <= adjacent_bin_index < self.calibration_bins:
                    bin_predictions = [
                        (conf, result, weight)
                        for conf, result, weight in history
                        if int(conf * self.calibration_bins / 100) == adjacent_bin_index
                    ]
                    if bin_predictions:
                        break

        if not bin_predictions:
            return raw_confidence

        # Calculate weighted accuracy for that bin
        total_weight = sum(w for _, _, w in bin_predictions)
        correct_weight = sum(w for _, result, w in bin_predictions if result)

        if total_weight == 0:
            return raw_confidence

        accuracy_in_bin = correct_weight / total_weight

        # Simple linear interpolation between raw confidence and historical accuracy
        # Give more weight to historical accuracy if more data is available
        data_points_weight = min(len(bin_predictions) / 50.0, 1.0) * 0.5
        calibrated = (
            raw_confidence * (1 - data_points_weight)
            + (accuracy_in_bin * 100) * data_points_weight
        )

        return np.clip(calibrated, 0, 100)
