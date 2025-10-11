import json
from pathlib import Path
from typing import Dict, Any

class SignalTracker:
    def __init__(self, storage_path: str = "signal_history.json"):
        self.storage_path = Path(storage_path)
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_history(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def record(self, signal_id: str, outcome: bool, details: Dict[str, Any]):
        """
        Records the outcome of a trading signal.
        
        :param signal_id: A unique identifier for the signal.
        :param outcome: True for a successful trade (profit), False otherwise.
        :param details: A dictionary containing signal parameters for later analysis.
        """
        self.history[signal_id] = {
            "outcome": outcome,
            "details": details
        }
        self._save_history()

    def update_adaptive_systems(self, analysis_engine):
        """
        Updates adaptive components like threshold managers and ML calibrators
        based on historical signal performance.
        """
        if not hasattr(analysis_engine, 'threshold_manager') or not hasattr(analysis_engine, 'ml_calibrator'):
            return

        for signal_id, data in self.history.items():
            details = data.get('details', {})
            outcome = data.get('outcome')

            # Update AdaptiveThresholdManager
            if all(k in details for k in ['volatility_regime', 'hurst_range', 'threshold']):
                analysis_engine.threshold_manager.record_performance(
                    details['volatility_regime'],
                    details['hurst_range'],
                    details['threshold'],
                    outcome
                )

            # Update MLConfidenceCalibrator
            if 'ml_predictions' in details:
                for model_name, pred_data in details['ml_predictions'].items():
                    confidence = pred_data.get('confidence')
                    if confidence is not None:
                        analysis_engine.ml_calibrator.add_prediction(
                            model_name,
                            confidence,
                            outcome
                        )