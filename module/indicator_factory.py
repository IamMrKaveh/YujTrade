from typing import Dict, Optional

from .indicators import TechnicalIndicator, get_all_indicators, IndicatorResult
from .logger_config import logger


class IndicatorFactory:
    def __init__(self):
        self._indicators = get_all_indicators()

    def create(self, name: str) -> Optional[TechnicalIndicator]:
        return self._indicators.get(name.lower())

    def get_all_indicator_names(self):
        return list(self._indicators.keys())