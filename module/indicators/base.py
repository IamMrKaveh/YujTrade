from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd

from ..core import IndicatorResult


class IndicatorInterface(Protocol):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ...


class TechnicalIndicator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        pass
