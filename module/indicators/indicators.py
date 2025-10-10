from typing import Dict

from .base import TechnicalIndicator
from .custom import CorrelationCoefficientIndicator, ElderRayIndexIndicator, FractalIndicator, LiquidityLevelsIndicator, MedianPriceIndicator, PivotPointsIndicator, PriceActionPatternIndicator, SmartMoneyConceptIndicator, TypicalPriceIndicator, WeightedClosePriceIndicator, WyckoffVolumeSpreadIndicator # type: ignore
from .cycle import HilbertDominantCycleIndicator, HilbertTrendVsCycleModeIndicator, KSTIndicator, KaufmanEfficiencyRatioIndicator, VortexIndicator # type: ignore
from .momentum import AwesomeOscillatorIndicator, CCIIndicator, ChandeMomentumOscillatorIndicator, ConnorsRSIIndicator, CoppockCurveIndicator, DetrendedPriceOscillatorIndicator, FisherTransformIndicator, MomentumIndicator, PPOIndicator, QQEIndicator, RMIIndicator, ROCIndicator, RSI2Indicator, RSIIndicator, RelativeVigorIndexIndicator, SchaffTrendCycleIndicator, SqueezeMomentumIndicator, StochRSIIndicator, StochasticIndicator, StochasticMomentumIndexIndicator, TRIXIndicator, TSIIndicator, UltimateOscillatorIndicator, WilliamsRIndicator # type: ignore
from .trend import ADXIndicator, AroonIndicator, DEMAIndicator, FRAMAIndicator, GannHiLoActivatorIndicator, HullMovingAverageIndicator, IchimokuIndicator, KAMAIndicator, LinearRegressionIndicator, LinearRegressionSlopeIndicator, MACDIndicator, MAMAIndicator, MarketStructureIndicator, MovingAverageIndicator, MovingAverageRibbonIndicator, ParabolicSARIndicator, SuperTrendIndicator, T3Indicator, TEMAIndicator, TrendIntensityIndexIndicator, VIDYAIndicator, ZLEMAIndicator # type: ignore
from .volatility import ATRBandsIndicator, ATRIndicator, BollingerBandsIndicator, BollingerBandwidthIndicator, ChaikinVolatilityIndicator, ChoppinessIndexIndicator, DonchianChannelsIndicator, HistoricalVolatilityIndicator, KeltnerChannelsIndicator, MassIndexIndicator, StandardDeviationIndicator, UlcerIndexIndicator # type: ignore
from .volume import ADLineIndicator, AccumulationDistributionIndexIndicator, AccumulationDistributionOscillatorIndicator, BalanceOfPowerIndicator, BalanceOfPowerRSIIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, ForceIndexIndicator, KlingerVolumeOscillatorIndicator, MFIBillWilliamsIndicator, MoneyFlowIndexIndicator, NVIIndicator, OBVIndicator, OrderFlowImbalanceIndicator, PVIIndicator, PVOIndicator, PriceVolumeRankIndicator, PriceVolumeTrendIndicator, VWAPIndicator, VWMAIndicator, VolumeIndicator, VolumeOscillatorIndicator, VolumeWeightedRSIIndicator # type: ignore


def get_all_indicators() -> Dict[str, TechnicalIndicator]:
    return {
        'sma_20': MovingAverageIndicator(20, "sma"), 'sma_50': MovingAverageIndicator(50, "sma"),
        'ema_12': MovingAverageIndicator(12, "ema"), 'ema_26': MovingAverageIndicator(26, "ema"),
        'rsi': RSIIndicator(), 'macd': MACDIndicator(), 'bb': BollingerBandsIndicator(),
        'stoch': StochasticIndicator(), 'volume': VolumeIndicator(), 'atr': ATRIndicator(),
        'ichimoku': IchimokuIndicator(), 'williams_r': WilliamsRIndicator(), 'cci': CCIIndicator(),
        'supertrend': SuperTrendIndicator(), 'adx': ADXIndicator(), 'cmf': ChaikinMoneyFlowIndicator(),
        'obv': OBVIndicator(), 'squeeze': SqueezeMomentumIndicator(), 'psar': ParabolicSARIndicator(),
        'vwap': VWAPIndicator(), 'mfi': MoneyFlowIndexIndicator(), 'aroon': AroonIndicator(),
        'uo': UltimateOscillatorIndicator(), 'roc': ROCIndicator(), 'ad_line': ADLineIndicator(),
        'force_index': ForceIndexIndicator(), 'vwma': VWMAIndicator(), 'keltner': KeltnerChannelsIndicator(),
        'donchian': DonchianChannelsIndicator(), 'trix': TRIXIndicator(), 'eom': EaseOfMovementIndicator(),
        'std_dev': StandardDeviationIndicator(), 'stochrsi': StochRSIIndicator(),
        'kst': KSTIndicator(), 'mass': MassIndexIndicator(), 'corr_coef': CorrelationCoefficientIndicator(),
        'elder_ray': ElderRayIndexIndicator(), 'pivot': PivotPointsIndicator(), 'momentum': MomentumIndicator(),
        'dpo': DetrendedPriceOscillatorIndicator(), 'choppiness': ChoppinessIndexIndicator(),
        'vortex': VortexIndicator(), 'awesome': AwesomeOscillatorIndicator(), 'cmo': ChandeMomentumOscillatorIndicator(),
        'rvi': RelativeVigorIndexIndicator(), 'pvr': PriceVolumeRankIndicator(), 
        'ado': AccumulationDistributionOscillatorIndicator(), 'pvt': PriceVolumeTrendIndicator(),
        'bop': BalanceOfPowerIndicator(), 'linreg': LinearRegressionIndicator(), 
        'linreg_slope': LinearRegressionSlopeIndicator(), 'median_price': MedianPriceIndicator(),
        'typical_price': TypicalPriceIndicator(), 'weighted_close': WeightedClosePriceIndicator(),
        'hma': HullMovingAverageIndicator(), 'zlema': ZLEMAIndicator(), 'kama': KAMAIndicator(),
        't3': T3Indicator(), 'dema': DEMAIndicator(), 'tema': TEMAIndicator(),
        'fisher': FisherTransformIndicator(), 'stc': SchaffTrendCycleIndicator(),
        'qqe': QQEIndicator(), 'connors_rsi': ConnorsRSIIndicator(), 'smi': StochasticMomentumIndexIndicator(),
        'tsi': TSIIndicator(), 'gann_hilo': GannHiLoActivatorIndicator(), 'ma_ribbon': MovingAverageRibbonIndicator(),
        'fractal': FractalIndicator(), 'chaikin_vol': ChaikinVolatilityIndicator(), 
        'historical_vol': HistoricalVolatilityIndicator(), 'ulcer_index': UlcerIndexIndicator(),
        'atr_bands': ATRBandsIndicator(), 'bbw': BollingerBandwidthIndicator(),
        'volume_osc': VolumeOscillatorIndicator(), 'kvo': KlingerVolumeOscillatorIndicator(),
        'frama': FRAMAIndicator(), 'vidya': VIDYAIndicator(), 'mama': MAMAIndicator(),
        'rmi': RMIIndicator(), 'rsi2': RSI2Indicator(), 'ppo': PPOIndicator(), 'pvo': PVOIndicator(),
        'nvi': NVIIndicator(), 'pvi': PVIIndicator(), 'mfi_bw': MFIBillWilliamsIndicator(),
        'ht_dc': HilbertDominantCycleIndicator(), 'ht_trend_mode': HilbertTrendVsCycleModeIndicator(),
        'er': KaufmanEfficiencyRatioIndicator(), 'coppock': CoppockCurveIndicator(),
        'bop_rsi': BalanceOfPowerRSIIndicator(), 'price_action': PriceActionPatternIndicator(),
        'market_structure': MarketStructureIndicator(), 'liquidity_levels': LiquidityLevelsIndicator(),
        'vw_rsi': VolumeWeightedRSIIndicator(), 'smc': SmartMoneyConceptIndicator(),
        'wyckoff_vsa': WyckoffVolumeSpreadIndicator(), 'adi': AccumulationDistributionIndexIndicator(),
        'tii': TrendIntensityIndexIndicator(), 'order_flow': OrderFlowImbalanceIndicator()
    }