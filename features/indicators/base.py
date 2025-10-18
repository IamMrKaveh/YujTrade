from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd

from common.core import IndicatorResult
from common.exceptions import IndicatorError, InsufficientDataError


class IndicatorInterface(Protocol):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult: ...


class TechnicalIndicator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculates the indicator value.

        Raises:
            InsufficientDataError: If the provided data is not enough for calculation.
            IndicatorError: For other calculation-related errors.
        """
        pass
