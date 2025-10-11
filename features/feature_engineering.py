import pandas as pd
from typing import Dict, Any, List

from features.indicators.factory import IndicatorFactory
from config.settings import ConfigManager
from common.exceptions import IndicatorError
from config.logger import logger


class FeatureEngineer:
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager if config_manager else ConfigManager()
        self.indicator_factory = IndicatorFactory(self.config_manager)
        self.active_indicators = self.config_manager.get_indicator_configs()

    def create_features(self, data: pd.DataFrame, indicators_to_run: List[str] = None) -> pd.DataFrame:
        """
        Adds multiple indicator features to the dataframe.
        """
        if indicators_to_run is None:
            indicators_to_run = [ind['name'] for ind in self.active_indicators]

        features_df = data.copy()
        
        for indicator_name in indicators_to_run:
            try:
                indicator_instance = self.indicator_factory.get_indicator(indicator_name)
                if indicator_instance:
                    # The indicator's process method now returns a DataFrame
                    indicator_output = indicator_instance.process(features_df)
                    
                    # Merge the output with the main features DataFrame
                    if not indicator_output.empty:
                        features_df = pd.merge(features_df, indicator_output, left_index=True, right_index=True, how='left')

            except IndicatorError as e:
                logger.warning(f"Could not calculate indicator '{indicator_name}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error with indicator '{indicator_name}': {e}", exc_info=True)
        
        return features_df

    def get_last_indicator_results(self) -> Dict[str, Any]:
        """
        Retrieves the last calculated results from the IndicatorFactory.
        """
        return self.indicator_factory.last_results