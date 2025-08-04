# models/market.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from decimal import Decimal
from datetime import datetime, timedelta
from .base import BaseModel

class MarketCondition(Enum):
    """وضعیت بازار"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CALM = "calm"
    EXTREMELY_BULLISH = "extremely_bullish"
    EXTREMELY_BEARISH = "extremely_bearish"

class TrendDirection(Enum):
    """جهت روند"""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"

class MarketPhase(Enum):
    """فاز بازار"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    DECLINE = "decline"

class VolumeProfile(Enum):
    """پروفایل حجم معاملات"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREMELY_HIGH = "extremely_high"

@dataclass
class MarketIndicators:
    """اندیکاتورهای تکنیکال بازار"""
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    moving_average_50: Optional[Decimal] = None
    moving_average_200: Optional[Decimal] = None
    bollinger_upper: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    support_level: Optional[Decimal] = None
    resistance_level: Optional[Decimal] = None
    fibonacci_levels: List[Decimal] = field(default_factory=list)

@dataclass
class MarketSentiment:
    """احساسات بازار"""
    fear_greed_index: Optional[int] = None
    social_sentiment_score: Optional[float] = None
    news_sentiment: Optional[str] = None
    whale_activity: Optional[str] = None
    institutional_flow: Optional[str] = None
    retail_sentiment: Optional[str] = None

@dataclass
class VolumeAnalysis:
    """تحلیل حجم معاملات"""
    total_volume_24h: Optional[Decimal] = None
    volume_change_24h: Optional[float] = None
    volume_profile: VolumeProfile = VolumeProfile.NORMAL
    top_volume_pairs: List[str] = field(default_factory=list)
    volume_weighted_price: Optional[Decimal] = None

@dataclass
class MarketCorrelations:
    """همبستگی‌های بازار"""
    btc_correlation: Optional[float] = None
    eth_correlation: Optional[float] = None
    traditional_markets_correlation: Optional[float] = None
    gold_correlation: Optional[float] = None
    dollar_index_correlation: Optional[float] = None

@dataclass
class MarketMetrics:
    """متریک‌های پیشرفته بازار"""
    market_cap_to_realized_value: Optional[float] = None
    network_value_to_transactions: Optional[float] = None
    realized_volatility: Optional[float] = None
    implied_volatility: Optional[float] = None
    funding_rates_average: Optional[float] = None
    open_interest: Optional[Decimal] = None
    long_short_ratio: Optional[float] = None

@dataclass
class MarketData(BaseModel):
    """داده‌های کلی بازار"""
    # اطلاعات اصلی
    total_market_cap: Optional[Decimal] = None
    total_volume_24h: Optional[Decimal] = None
    btc_dominance: Optional[float] = None
    eth_dominance: Optional[float] = None
    altcoin_market_cap: Optional[Decimal] = None
    
    # وضعیت و روند
    condition: MarketCondition = MarketCondition.SIDEWAYS
    trend_direction: TrendDirection = TrendDirection.NEUTRAL
    market_phase: MarketPhase = MarketPhase.ACCUMULATION
    volatility_score: float = 0.0
    trend_strength: float = 0.0
    
    # تغییرات درصدی
    market_cap_change_24h: Optional[float] = None
    market_cap_change_7d: Optional[float] = None
    market_cap_change_30d: Optional[float] = None
    
    # اندیکاتورها و احساسات
    indicators: MarketIndicators = field(default_factory=MarketIndicators)
    sentiment: MarketSentiment = field(default_factory=MarketSentiment)
    volume_analysis: VolumeAnalysis = field(default_factory=VolumeAnalysis)
    correlations: MarketCorrelations = field(default_factory=MarketCorrelations)
    metrics: MarketMetrics = field(default_factory=MarketMetrics)
    
    # اطلاعات زمانی
    timestamp: datetime = field(default_factory=datetime.now)
    last_updated: Optional[datetime] = None
    data_freshness_minutes: int = 0
    
    # آمار بازار
    active_cryptocurrencies: Optional[int] = None
    active_markets: Optional[int] = None
    top_gainers: List[str] = field(default_factory=list)
    top_losers: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """ایجاد نمونه از دیکشنری"""
        indicators_data = data.get('indicators', {})
        sentiment_data = data.get('sentiment', {})
        volume_data = data.get('volume_analysis', {})
        correlations_data = data.get('correlations', {})
        metrics_data = data.get('metrics', {})
        
        return cls(
            # اطلاعات اصلی
            total_market_cap=Decimal(str(data['total_market_cap'])) if data.get('total_market_cap') else None,
            total_volume_24h=Decimal(str(data['total_volume_24h'])) if data.get('total_volume_24h') else None,
            btc_dominance=data.get('btc_dominance'),
            eth_dominance=data.get('eth_dominance'),
            altcoin_market_cap=Decimal(str(data['altcoin_market_cap'])) if data.get('altcoin_market_cap') else None,
            
            # وضعیت و روند
            condition=MarketCondition(data.get('condition', 'sideways')),
            trend_direction=TrendDirection(data.get('trend_direction', 'neutral')),
            market_phase=MarketPhase(data.get('market_phase', 'accumulation')),
            volatility_score=data.get('volatility_score', 0.0),
            trend_strength=data.get('trend_strength', 0.0),
            
            # تغییرات درصدی
            market_cap_change_24h=data.get('market_cap_change_24h'),
            market_cap_change_7d=data.get('market_cap_change_7d'),
            market_cap_change_30d=data.get('market_cap_change_30d'),
            
            # اندیکاتورها
            indicators=MarketIndicators(
                rsi=indicators_data.get('rsi'),
                macd_signal=indicators_data.get('macd_signal'),
                moving_average_50=Decimal(str(indicators_data['moving_average_50'])) if indicators_data.get('moving_average_50') else None,
                moving_average_200=Decimal(str(indicators_data['moving_average_200'])) if indicators_data.get('moving_average_200') else None,
                bollinger_upper=Decimal(str(indicators_data['bollinger_upper'])) if indicators_data.get('bollinger_upper') else None,
                bollinger_lower=Decimal(str(indicators_data['bollinger_lower'])) if indicators_data.get('bollinger_lower') else None,
                support_level=Decimal(str(indicators_data['support_level'])) if indicators_data.get('support_level') else None,
                resistance_level=Decimal(str(indicators_data['resistance_level'])) if indicators_data.get('resistance_level') else None,
                fibonacci_levels=[Decimal(str(level)) for level in indicators_data.get('fibonacci_levels', [])]
            ),
            
            # احساسات
            sentiment=MarketSentiment(
                fear_greed_index=sentiment_data.get('fear_greed_index'),
                social_sentiment_score=sentiment_data.get('social_sentiment_score'),
                news_sentiment=sentiment_data.get('news_sentiment'),
                whale_activity=sentiment_data.get('whale_activity'),
                institutional_flow=sentiment_data.get('institutional_flow'),
                retail_sentiment=sentiment_data.get('retail_sentiment')
            ),
            
            # تحلیل حجم
            volume_analysis=VolumeAnalysis(
                total_volume_24h=Decimal(str(volume_data['total_volume_24h'])) if volume_data.get('total_volume_24h') else None,
                volume_change_24h=volume_data.get('volume_change_24h'),
                volume_profile=VolumeProfile(volume_data.get('volume_profile', 'normal')),
                top_volume_pairs=volume_data.get('top_volume_pairs', []),
                volume_weighted_price=Decimal(str(volume_data['volume_weighted_price'])) if volume_data.get('volume_weighted_price') else None
            ),
            
            # همبستگی‌ها
            correlations=MarketCorrelations(
                btc_correlation=correlations_data.get('btc_correlation'),
                eth_correlation=correlations_data.get('eth_correlation'),
                traditional_markets_correlation=correlations_data.get('traditional_markets_correlation'),
                gold_correlation=correlations_data.get('gold_correlation'),
                dollar_index_correlation=correlations_data.get('dollar_index_correlation')
            ),
            
            # متریک‌های پیشرفته
            metrics=MarketMetrics(
                market_cap_to_realized_value=metrics_data.get('market_cap_to_realized_value'),
                network_value_to_transactions=metrics_data.get('network_value_to_transactions'),
                realized_volatility=metrics_data.get('realized_volatility'),
                implied_volatility=metrics_data.get('implied_volatility'),
                funding_rates_average=metrics_data.get('funding_rates_average'),
                open_interest=Decimal(str(metrics_data['open_interest'])) if metrics_data.get('open_interest') else None,
                long_short_ratio=metrics_data.get('long_short_ratio')
            ),
            
            # اطلاعات زمانی
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now(),
            last_updated=datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else None,
            data_freshness_minutes=data.get('data_freshness_minutes', 0),
            
            # آمار بازار
            active_cryptocurrencies=data.get('active_cryptocurrencies'),
            active_markets=data.get('active_markets'),
            top_gainers=data.get('top_gainers', []),
            top_losers=data.get('top_losers', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        return {
            # اطلاعات اصلی
            'total_market_cap': str(self.total_market_cap) if self.total_market_cap else None,
            'total_volume_24h': str(self.total_volume_24h) if self.total_volume_24h else None,
            'btc_dominance': self.btc_dominance,
            'eth_dominance': self.eth_dominance,
            'altcoin_market_cap': str(self.altcoin_market_cap) if self.altcoin_market_cap else None,
            
            # وضعیت و روند
            'condition': self.condition.value,
            'trend_direction': self.trend_direction.value,
            'market_phase': self.market_phase.value,
            'volatility_score': self.volatility_score,
            'trend_strength': self.trend_strength,
            
            # تغییرات درصدی
            'market_cap_change_24h': self.market_cap_change_24h,
            'market_cap_change_7d': self.market_cap_change_7d,
            'market_cap_change_30d': self.market_cap_change_30d,
            
            # اندیکاتورها
            'indicators': {
                'rsi': self.indicators.rsi,
                'macd_signal': self.indicators.macd_signal,
                'moving_average_50': str(self.indicators.moving_average_50) if self.indicators.moving_average_50 else None,
                'moving_average_200': str(self.indicators.moving_average_200) if self.indicators.moving_average_200 else None,
                'bollinger_upper': str(self.indicators.bollinger_upper) if self.indicators.bollinger_upper else None,
                'bollinger_lower': str(self.indicators.bollinger_lower) if self.indicators.bollinger_lower else None,
                'support_level': str(self.indicators.support_level) if self.indicators.support_level else None,
                'resistance_level': str(self.indicators.resistance_level) if self.indicators.resistance_level else None,
                'fibonacci_levels': [str(level) for level in self.indicators.fibonacci_levels]
            },
            
            # احساسات
            'sentiment': {
                'fear_greed_index': self.sentiment.fear_greed_index,
                'social_sentiment_score': self.sentiment.social_sentiment_score,
                'news_sentiment': self.sentiment.news_sentiment,
                'whale_activity': self.sentiment.whale_activity,
                'institutional_flow': self.sentiment.institutional_flow,
                'retail_sentiment': self.sentiment.retail_sentiment
            },
            
            # تحلیل حجم
            'volume_analysis': {
                'total_volume_24h': str(self.volume_analysis.total_volume_24h) if self.volume_analysis.total_volume_24h else None,
                'volume_change_24h': self.volume_analysis.volume_change_24h,
                'volume_profile': self.volume_analysis.volume_profile.value,
                'top_volume_pairs': self.volume_analysis.top_volume_pairs,
                'volume_weighted_price': str(self.volume_analysis.volume_weighted_price) if self.volume_analysis.volume_weighted_price else None
            },
            
            # همبستگی‌ها
            'correlations': {
                'btc_correlation': self.correlations.btc_correlation,
                'eth_correlation': self.correlations.eth_correlation,
                'traditional_markets_correlation': self.correlations.traditional_markets_correlation,
                'gold_correlation': self.correlations.gold_correlation,
                'dollar_index_correlation': self.correlations.dollar_index_correlation
            },
            
            # متریک‌های پیشرفته
            'metrics': {
                'market_cap_to_realized_value': self.metrics.market_cap_to_realized_value,
                'network_value_to_transactions': self.metrics.network_value_to_transactions,
                'realized_volatility': self.metrics.realized_volatility,
                'implied_volatility': self.metrics.implied_volatility,
                'funding_rates_average': self.metrics.funding_rates_average,
                'open_interest': str(self.metrics.open_interest) if self.metrics.open_interest else None,
                'long_short_ratio': self.metrics.long_short_ratio
            },
            
            # اطلاعات زمانی
            'timestamp': self.timestamp.isoformat(),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'data_freshness_minutes': self.data_freshness_minutes,
            
            # آمار بازار
            'active_cryptocurrencies': self.active_cryptocurrencies,
            'active_markets': self.active_markets,
            'top_gainers': self.top_gainers,
            'top_losers': self.top_losers
        }
    
    def is_bullish(self) -> bool:
        """بررسی صعودی بودن بازار"""
        return self.condition in [MarketCondition.BULLISH, MarketCondition.EXTREMELY_BULLISH]
    
    def is_bearish(self) -> bool:
        """بررسی نزولی بودن بازار"""
        return self.condition in [MarketCondition.BEARISH, MarketCondition.EXTREMELY_BEARISH]
    
    def is_volatile(self) -> bool:
        """بررسی پرنوسان بودن بازار"""
        return self.condition == MarketCondition.VOLATILE or self.volatility_score > 70
    
    def get_market_summary(self) -> str:
        """خلاصه وضعیت بازار"""
        summary = f"Market Cap: ${self.total_market_cap:,.0f}" if self.total_market_cap else "Market Cap: N/A"
        summary += f"\nBTC Dominance: {self.btc_dominance:.1f}%" if self.btc_dominance else "\nBTC Dominance: N/A"
        summary += f"\nCondition: {self.condition.value.title()}"
        summary += f"\nTrend: {self.trend_direction.value.title()}"
        summary += f"\nVolatility: {self.volatility_score:.1f}/100"
        
        if self.sentiment.fear_greed_index:
            summary += f"\nFear & Greed: {self.sentiment.fear_greed_index}/100"
        
        return summary
    
    def is_data_fresh(self, max_age_minutes: int = 15) -> bool:
        """بررسی تازگی داده‌ها"""
        return self.data_freshness_minutes <= max_age_minutes
    
    def get_dominant_trend(self) -> str:
        """تعیین روند غالب"""
        if self.trend_strength > 70:
            return f"Strong {self.trend_direction.value.title()}"
        elif self.trend_strength > 40:
            return f"Moderate {self.trend_direction.value.title()}"
        else:
            return "Weak/Sideways"
    
    def calculate_market_health_score(self) -> float:
        """محاسبه امتیاز سلامت بازار"""
        score = 50.0  # نقطه شروع
        
        # بر اساس احساسات
        if self.sentiment.fear_greed_index:
            if 40 <= self.sentiment.fear_greed_index <= 60:
                score += 10  # تعادل خوب
            elif self.sentiment.fear_greed_index < 20 or self.sentiment.fear_greed_index > 80:
                score -= 15  # افراط
        
        # بر اساس نوسانات
        if self.volatility_score < 30:
            score += 10  # نوسانات کم
        elif self.volatility_score > 70:
            score -= 10  # نوسانات زیاد
        
        # بر اساس حجم معاملات
        if self.volume_analysis.volume_change_24h:
            if self.volume_analysis.volume_change_24h > 20:
                score += 5  # افزایش حجم
            elif self.volume_analysis.volume_change_24h < -20:
                score -= 5  # کاهش حجم
        
        # بر اساس قدرت روند
        if self.trend_strength > 60:
            score += 15  # روند قوی
        elif self.trend_strength < 20:
            score -= 10  # روند ضعیف
        
        return max(0, min(100, score))  # محدود کردن بین 0 تا 100