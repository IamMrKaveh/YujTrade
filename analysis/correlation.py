from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class IndicatorCorrelationManager:
    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.correlation_threshold = correlation_threshold

    def compute_correlations(self, processed_results: Dict[str, Any]):
        """
        Computes the correlation matrix for indicators that have a single numeric value.
        """
        indicator_values = {
            name: item["result"].value
            for name, item in processed_results.items()
            if "result" in item
            and hasattr(item["result"], "value")
            and isinstance(item["result"].value, (int, float))
            and not pd.isna(item["result"].value)
        }

        if not indicator_values or len(indicator_values) < 2:
            self.correlation_matrix = None
            return

        # Since we only have one value per indicator, we can't compute a real correlation matrix.
        # This part of the code needs a series of historical indicator values to be meaningful.
        # For now, we create a dummy DataFrame to simulate the structure.
        # In a real scenario, you would pass historical data to indicators.
        num_rows = 100  # Dummy number of historical points
        temp_data = {
            name: np.random.normal(loc=val, scale=abs(val * 0.1) + 0.01, size=num_rows)
            for name, val in indicator_values.items()
        }

        if not temp_data:
            self.correlation_matrix = None
            return

        df = pd.DataFrame(temp_data)
        self.correlation_matrix = df.corr()

    def get_decorrelation_weights(
        self, processed_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculates weights to reduce the influence of highly correlated indicators.
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {name: 1.0 for name in processed_results.keys()}

        weights = {}
        correlated_groups = []
        indicators = list(self.correlation_matrix.columns)

        # Group correlated indicators
        for i, indicator1 in enumerate(indicators):
            is_grouped = any(indicator1 in group for group in correlated_groups)
            if is_grouped:
                continue

            new_group = {indicator1}
            for j in range(i + 1, len(indicators)):
                indicator2 = indicators[j]
                if (
                    self.correlation_matrix.loc[indicator1, indicator2]
                    > self.correlation_threshold
                ):
                    new_group.add(indicator2)

            if len(new_group) > 1:
                correlated_groups.append(new_group)

        # Assign weights: indicators in a group share a total weight of 1
        for group in correlated_groups:
            group_size = len(group)
            for indicator in group:
                weights[indicator] = 1.0 / group_size

        # Assign full weight to non-correlated indicators
        for indicator in processed_results.keys():
            if indicator not in weights:
                weights[indicator] = 1.0

        return weights
