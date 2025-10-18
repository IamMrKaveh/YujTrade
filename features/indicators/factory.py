from typing import Optional, List, Dict, Type

from features.indicators.all_indicators import get_all_indicators
from features.indicators.base import TechnicalIndicator
from features.indicators.trend import MovingAverageIndicator


class IndicatorFactory:
    def __init__(self):
        self._indicator_classes: Dict[str, Type[TechnicalIndicator]] = (
            get_all_indicators()
        )

    def create(self, name: str) -> Optional[TechnicalIndicator]:
        """
        Dynamically creates an instance of a technical indicator.
        """
        name_lower = name.lower()

        # Handle special cases like 'sma_20', 'ema_12'
        if name_lower.startswith(("sma_", "ema_")):
            try:
                ma_type, period_str = name_lower.split("_")
                period = int(period_str)
                return MovingAverageIndicator(period=period, ma_type=ma_type)
            except (ValueError, IndexError):
                return None  # Invalid format

        indicator_class = self._indicator_classes.get(name_lower)
        if indicator_class:
            return indicator_class()

        return None

    def get_all_indicator_names(self) -> List[str]:
        # This can be expanded to include dynamic names like 'sma_20' if needed,
        # but for now, returning the base names is sufficient.
        return list(self._indicator_classes.keys())
