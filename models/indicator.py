from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class IndicatorResult:
    name: str
    value: float
    signal_strength: float
    interpretation: str

class IndicatorInterface(Protocol):
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        ...

class TechnicalIndicator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        pass