import logging
import pandas as pd
import numpy as np

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Set up logger
logger = logging.getLogger(__name__)

def calculate_fibonacci_levels(df, lookback=100):
    try:
        if df is None or len(df) < lookback:
            return None
        
        recent_data = df.tail(lookback)
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        
        diff = high_price - low_price
        
        fib_levels = {
            'fib_0': high_price,
            'fib_236': high_price - (diff * 0.236),
            'fib_382': high_price - (diff * 0.382),
            'fib_500': high_price - (diff * 0.5),
            'fib_618': high_price - (diff * 0.618),
            'fib_786': high_price - (diff * 0.786),
            'fib_100': low_price
        }
        
        return fib_levels
    except Exception:
        return None

def calculate_parabolic_sar(df, af=0.02, max_af=0.2):
    try:
        if df is None or len(df) < 5:
            return None
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))
        af_val = np.zeros(len(df))
        ep = np.zeros(len(df))
        
        sar[0] = low[0]
        trend[0] = 1
        af_val[0] = af
        ep[0] = high[0]
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:
                sar[i] = sar[i-1] + af_val[i-1] * (ep[i-1] - sar[i-1])
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af_val[i] = af
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af_val[i] = min(af_val[i-1] + af, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af_val[i] = af_val[i-1]
            else:
                sar[i] = sar[i-1] - af_val[i-1] * (sar[i-1] - ep[i-1])
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af_val[i] = af
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af_val[i] = min(af_val[i-1] + af, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af_val[i] = af_val[i-1]
        
        return pd.Series(sar, index=df.index, name='psar')
    except Exception:
        return None

def calculate_ichimoku(df):
    try:
        if df is None or len(df) < 52:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    except Exception:
        return None

def calculate_money_flow_index(df, period=14):
    try:
        if df is None or len(df) < period + 1:
            return None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        price_diff = typical_price.diff()
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    except Exception:
        return None

def calculate_commodity_channel_index(df, period=20):
    try:
        if df is None or len(df) < period:
            return None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    except Exception:
        return None

def calculate_williams_r(df, period=14):
    try:
        if df is None or len(df) < period:
            return None
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return williams_r
    except Exception:
        return None

# اندیکاتورهای مومنتوم اضافی
def calculate_ultimate_oscillator(df, period1=7, period2=14, period3=28):
    """محاسبه Ultimate Oscillator"""
    try:
        if df is None or len(df) < max(period1, period2, period3):
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        # True Low = minimum of Low or previous Close
        true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
        
        # Buying Pressure = Close - True Low
        buying_pressure = close - true_low
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average calculations for 3 periods
        bp1 = buying_pressure.rolling(window=period1).sum()
        tr1_sum = true_range.rolling(window=period1).sum()
        
        bp2 = buying_pressure.rolling(window=period2).sum()
        tr2_sum = true_range.rolling(window=period2).sum()
        
        bp3 = buying_pressure.rolling(window=period3).sum()
        tr3_sum = true_range.rolling(window=period3).sum()
        
        # Ultimate Oscillator formula
        uo = 100 * (4 * (bp1 / tr1_sum) + 2 * (bp2 / tr2_sum) + (bp3 / tr3_sum)) / 7
        
        return uo
    except Exception as e:
        logger.warning(f"Error calculating Ultimate Oscillator: {e}")
        return None

def calculate_rate_of_change(df, period=14):
    """محاسبه Rate of Change (ROC)"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        
        return roc
    except Exception as e:
        logger.warning(f"Error calculating ROC: {e}")
        return None

def calculate_awesome_oscillator(df, fast_period=5, slow_period=34):
    """محاسبه Awesome Oscillator"""
    try:
        if df is None or len(df) < slow_period:
            return None
        
        median_price = (df['high'] + df['low']) / 2
        
        fast = median_price.rolling(window=fast_period).mean()
        slow = median_price.rolling(window=slow_period).mean()

        ao = fast - slow
        ao = ao.dropna()  # Remove NaN values
        
        if ao.empty:
            logger.warning("Awesome Oscillator calculation resulted in empty series")
            return None
        
        logger.info(f"Calculated Awesome Oscillator with {len(ao)} values")
        return ao
    except Exception as e:
        logger.warning(f"Error calculating Awesome Oscillator: {e}")
        return None

def calculate_trix(df, period=14):
    """محاسبه TRIX"""
    try:
        if df is None or len(df) < period * 3:
            return None
        
        close = df['close']
        
        # Triple smoothed EMA
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        # TRIX = Rate of change of triple smoothed EMA
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 10000
        
        return trix
    except Exception as e:
        logger.warning(f"Error calculating TRIX: {e}")
        return None

def calculate_dpo(df, period=20):
    """محاسبه Detrended Price Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        sma = close.rolling(window=period).mean()
        
        # DPO = Close - SMA shifted by (period/2 + 1)
        shift_period = int(period/2) + 1
        dpo = close - sma.shift(shift_period)
        
        dpo = dpo.dropna()  # Remove NaN values
        
        if dpo.empty:
            logger.warning("DPO calculation resulted in empty series")
            return None
        
        logger.info(f"Calculated DPO with {len(dpo)} values")
        
        return dpo
    except Exception as e:
        logger.warning(f"Error calculating DPO: {e}")
        return None

# ===== اندیکاتورهای حجم پیشرفته =====

def calculate_obv(df):
    """محاسبه On-Balance Volume"""
    try:
        if df is None or len(df) < 2:
            return None
        
        close = df['close']
        volume = df['volume']
        
        obv = []
        obv.append(0)  # مقدار اولیه
        
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[i-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[i-1] - volume.iloc[i])
            else:
                obv.append(obv[i-1])
        
        return pd.Series(obv, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating OBV: {e}")
        return None

def calculate_accumulation_distribution(df):
    """محاسبه Accumulation/Distribution Line"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        # Handle division by zero
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # A/D Line is cumulative sum of MFV
        ad_line = mfv.cumsum()
        
        return ad_line
    except Exception as e:
        logger.warning(f"Error calculating A/D Line: {e}")
        return None

def calculate_ad_line(df):
    """محاسبه Accumulation/Distribution Line"""
    try:
        if df is None or len(df) < 1:
            return None
            
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # محاسبه Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # در صورت صفر بودن دامنه
        
        # محاسبه Money Flow Volume
        mfv = clv * volume
        
        # محاسبه A/D Line تجمعی
        ad_line = mfv.cumsum()
        
        return ad_line
    except Exception:
        return None

def calculate_chaikin_money_flow(df, period=20):
    """محاسبه Chaikin Money Flow"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # CMF = Sum of MFV over period / Sum of Volume over period
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    except Exception as e:
        logger.warning(f"Error calculating CMF: {e}")
        return None

def calculate_volume_price_trend(df):
    """محاسبه Volume Price Trend"""
    try:
        if df is None or len(df) < 2:
            return None
        
        close = df['close']
        volume = df['volume']
        
        # Price change percentage
        price_change_pct = (close - close.shift(1)) / close.shift(1)
        
        # VPT = Previous VPT + Volume * Price Change %
        vpt = (price_change_pct * volume).cumsum()
        
        return vpt
    except Exception as e:
        logger.warning(f"Error calculating VPT: {e}")
        return None

def calculate_ease_of_movement(df, period=14):
    """محاسبه Ease of Movement"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Distance Moved
        distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        
        # Box Height
        box_height = (volume / 100000) / (high - low)
        
        # 1-Period EMV
        emv_1period = distance_moved / box_height
        emv_1period = emv_1period.replace([np.inf, -np.inf], 0).fillna(0)
        
        # EMV = SMA of 1-Period EMV
        emv = emv_1period.rolling(window=period).mean()
        
        return emv
    except Exception as e:
        logger.warning(f"Error calculating EMV: {e}")
        return None

# ===== اندیکاتورهای نوسان =====

def calculate_average_true_range(df, period=14):
    """محاسبه Average True Range"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return None

def calculate_atr(df, period=14):
    """محاسبه Average True Range"""
    try:
        if df is None or len(df) < period:
            return None
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    except Exception:
        return None

def calculate_keltner_channels(df, period=20, multiplier=2):
    """محاسبه Keltner Channels"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        atr = calculate_average_true_range(df, period)
        
        if atr is None:
            return None
        
        middle_line = close.rolling(window=period).mean()
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return {
            'keltner_upper': upper_channel,
            'keltner_middle': middle_line,
            'keltner_lower': lower_channel
        }
    except Exception as e:
        logger.warning(f"Error calculating Keltner Channels: {e}")
        return None

def calculate_donchian_channels(df, period=20):
    """محاسبه Donchian Channels"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'donchian_upper': upper_channel,
            'donchian_middle': middle_channel,
            'donchian_lower': lower_channel
        }
    except Exception as e:
        logger.warning(f"Error calculating Donchian Channels: {e}")
        return None

def calculate_standard_deviation(df, period=20):
    """محاسبه Standard Deviation"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        std_dev = close.rolling(window=period).std()
        
        return std_dev
    except Exception as e:
        logger.warning(f"Error calculating Standard Deviation: {e}")
        return None

def calculate_price_std(df, period=20):
    """محاسبه انحراف معیار قیمت"""
    try:
        if df is None or len(df) < period:
            return None
            
        close = df['close']
        std_dev = close.rolling(period).std()
        
        return std_dev
    except Exception:
        return None

# ===== اندیکاتورهای ترند پیشرفته =====

def calculate_supertrend(df, period=10, multiplier=3.0):
    """محاسبه Supertrend"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه ATR
        atr = calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # محاسبه Basic Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Final Bands
        final_upper_band = []
        final_lower_band = []
        supertrend = []
        direction = []
        
        for i in range(len(df)):
            if i == 0:
                final_upper_band.append(upper_band.iloc[i])
                final_lower_band.append(lower_band.iloc[i])
                supertrend.append(0)
                direction.append(1)
            else:
                # Final Upper Band
                if upper_band.iloc[i] < final_upper_band[i-1] or close.iloc[i-1] > final_upper_band[i-1]:
                    final_upper_band.append(upper_band.iloc[i])
                else:
                    final_upper_band.append(final_upper_band[i-1])
                
                # Final Lower Band
                if lower_band.iloc[i] > final_lower_band[i-1] or close.iloc[i-1] < final_lower_band[i-1]:
                    final_lower_band.append(lower_band.iloc[i])
                else:
                    final_lower_band.append(final_lower_band[i-1])
                
                # Direction and Supertrend
                if direction[i-1] == -1 and close.iloc[i] < final_lower_band[i]:
                    direction.append(-1)
                elif direction[i-1] == 1 and close.iloc[i] > final_upper_band[i]:
                    direction.append(1)
                elif direction[i-1] == -1 and close.iloc[i] >= final_lower_band[i]:
                    direction.append(1)
                elif direction[i-1] == 1 and close.iloc[i] <= final_upper_band[i]:
                    direction.append(-1)
                else:
                    direction.append(direction[i-1])
                
                if direction[i] == 1:
                    supertrend.append(final_lower_band[i])
                else:
                    supertrend.append(final_upper_band[i])
        
        return pd.Series(supertrend, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating Supertrend: {e}")
        return None

def calculate_aroon_oscillator(df, period=14):
    try:
        if df is None or len(df) < period:
            return None
        
        if 'high' not in df.columns or 'low' not in df.columns:
            return None
        
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        aroon_up = np.full(len(df), np.nan)
        aroon_down = np.full(len(df), np.nan)
        
        for i in range(period - 1, len(df)):
            high_period = high.iloc[i-period+1:i+1].values
            low_period = low.iloc[i-period+1:i+1].values
            
            high_max_idx = np.argmax(high_period)
            low_min_idx = np.argmin(low_period)
            
            periods_since_high = period - 1 - high_max_idx
            periods_since_low = period - 1 - low_min_idx
            
            aroon_up[i] = ((period - periods_since_high) / period) * 100
            aroon_down[i] = ((period - periods_since_low) / period) * 100
        
        aroon_up_series = pd.Series(aroon_up, index=df.index)
        aroon_down_series = pd.Series(aroon_down, index=df.index)
        aroon_oscillator = aroon_up_series - aroon_down_series
        
        return {
            'aroon_up': aroon_up_series,
            'aroon_down': aroon_down_series,
            'aroon_oscillator': aroon_oscillator
        }
    except Exception:
        return None

def calculate_aroon(df, period=14):
    """محاسبه Aroon Oscillator"""
    try:
        if df is None or len(df) < period:
            return None
            
        high = df['high']
        low = df['low']
        
        # پیدا کردن موقعیت بالاترین و پایین‌ترین قیمت
        aroon_up = ((period - high.rolling(period).apply(lambda x: period - 1 - x.argmax())) / period) * 100
        aroon_down = ((period - low.rolling(period).apply(lambda x: period - 1 - x.argmin())) / period) * 100
        
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    except Exception:
        return None

def calculate_adx(df, period=14):
    """محاسبه Average Directional Index (ADX)"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        
        # True Range
        atr = calculate_average_true_range(df, period)
        if atr is None:
            return None
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)
        
        plus_dm[up_move > down_move] = up_move[up_move > down_move]
        plus_dm[plus_dm < 0] = 0
        
        minus_dm[down_move > up_move] = down_move[down_move > up_move]
        minus_dm[minus_dm < 0] = 0
        
        # Smoothed DM
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # DI calculations
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        adx = dx.rolling(window=period).mean()
        
        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx
        }
    except Exception as e:
        logger.warning(f"Error calculating ADX: {e}")
        return None

def calculate_kama(df, period=10, fast_sc=2, slow_sc=30):
    """محاسبه Kaufman Adaptive Moving Average"""
    try:
        if df is None or len(df) < period:
            return None
        
        close = df['close']
        
        # Efficiency Ratio
        change = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(window=period).sum()
        er = change / volatility
        er = er.fillna(0)
        
        # Smoothing Constants
        fastest_sc = 2.0 / (fast_sc + 1)
        slowest_sc = 2.0 / (slow_sc + 1)
        sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # KAMA calculation
        kama = []
        kama.append(close.iloc[0])  # First value
        
        for i in range(1, len(close)):
            kama.append(kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1]))
        
        return pd.Series(kama, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating KAMA: {e}")
        return None

# الگوهای کندل استیک
def detect_hammer_doji_patterns(df):
    """تشخیص الگوهای Hammer و Doji"""
    try:
        if df is None or len(df) < 3:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه اجزای کندل
        body = abs(close - open_price)
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        total_range = high - low
        
        patterns = pd.DataFrame(index=df.index)
        
        # Hammer Pattern
        hammer_condition = (
            (lower_shadow >= 2 * body) &
            (upper_shadow <= 0.1 * total_range) &
            (body <= 0.3 * total_range)
        )
        patterns['hammer'] = hammer_condition
        
        # Doji Pattern
        doji_condition = (body <= 0.1 * total_range)
        patterns['doji'] = doji_condition
        
        # Shooting Star Pattern
        shooting_star_condition = (
            (upper_shadow >= 2 * body) &
            (lower_shadow <= 0.1 * total_range) &
            (body <= 0.3 * total_range)
        )
        patterns['shooting_star'] = shooting_star_condition
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Hammer/Doji patterns: {e}")
        return None

def detect_engulfing_patterns(df):
    """تشخیص الگوهای Engulfing"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['bullish_engulfing'] = False
        patterns['bearish_engulfing'] = False
        
        for i in range(1, len(df)):
            prev_open = open_price.iloc[i-1]
            prev_close = close.iloc[i-1]
            curr_open = open_price.iloc[i]
            curr_close = close.iloc[i]
            
            # Bullish Engulfing
            if (prev_close < prev_open and  # Previous red candle
                curr_close > curr_open and  # Current green candle
                curr_open < prev_close and  # Current opens below previous close
                curr_close > prev_open):    # Current closes above previous open
                patterns.iloc[i, patterns.columns.get_loc('bullish_engulfing')] = True
            
            # Bearish Engulfing
            if (prev_close > prev_open and  # Previous green candle
                curr_close < curr_open and  # Current red candle
                curr_open > prev_close and  # Current opens above previous close
                curr_close < prev_open):    # Current closes below previous open
                patterns.iloc[i, patterns.columns.get_loc('bearish_engulfing')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Engulfing patterns: {e}")
        return None

def detect_star_patterns(df):
    """تشخیص الگوهای Morning/Evening Star"""
    try:
        if df is None or len(df) < 3:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['morning_star'] = False
        patterns['evening_star'] = False
        
        for i in range(2, len(df)):
            # Morning Star Pattern
            first_red = close.iloc[i-2] < open_price.iloc[i-2]  # First candle is red
            small_body = abs(close.iloc[i-1] - open_price.iloc[i-1]) < abs(close.iloc[i-2] - open_price.iloc[i-2]) * 0.3  # Small middle candle
            gap_down = high.iloc[i-1] < low.iloc[i-2]  # Gap down
            third_green = close.iloc[i] > open_price.iloc[i]  # Third candle is green
            closes_into_first = close.iloc[i] > (open_price.iloc[i-2] + close.iloc[i-2]) / 2  # Closes well into first candle
            
            if first_red and small_body and gap_down and third_green and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('morning_star')] = True
            
            # Evening Star Pattern
            first_green = close.iloc[i-2] > open_price.iloc[i-2]  # First candle is green
            gap_up = low.iloc[i-1] > high.iloc[i-2]  # Gap up
            third_red = close.iloc[i] < open_price.iloc[i]  # Third candle is red
            closes_into_first = close.iloc[i] < (open_price.iloc[i-2] + close.iloc[i-2]) / 2  # Closes well into first candle
            
            if first_green and small_body and gap_up and third_red and closes_into_first:
                patterns.iloc[i, patterns.columns.get_loc('evening_star')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Star patterns: {e}")
        return None

def detect_morning_evening_star(df):
    """تشخیص الگوهای Morning/Evening Star"""
    try:
        if df is None or len(df) < 3:
            return None
            
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # محاسبه بدنه کندل‌ها
        body = abs(close - open_price)
        body_1 = body.shift(1)
        body_2 = body.shift(2)
        
        # Morning Star Pattern
        morning_star = ((close.shift(2) < open_price.shift(2)) &  # کندل نزولی
                       (body_1 < body_2 * 0.3) &  # کندل کوچک میانی
                       (close > open_price) &  # کندل صعودی
                       (close > (close.shift(2) + open_price.shift(2)) / 2))
        
        # Evening Star Pattern
        evening_star = ((close.shift(2) > open_price.shift(2)) &  # کندل صعودی
                       (body_1 < body_2 * 0.3) &  # کندل کوچک میانی
                       (close < open_price) &  # کندل نزولی
                       (close < (close.shift(2) + open_price.shift(2)) / 2))
        
        return {
            'morning_star': morning_star,
            'evening_star': evening_star
        }
    except Exception:
        return None

# اندیکاتورهای مارکت استراکچر
def calculate_pivot_points(df):
    """محاسبه Pivot Points"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Standard Pivot Points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    except Exception as e:
        logger.warning(f"Error calculating Pivot Points: {e}")
        return None

def calculate_support_resistance(df, window=20):
    """محاسبه سطوح Support و Resistance"""
    try:
        if df is None or len(df) < window:
            return None
        
        high = df['high']
        low = df['low']
        
        # Local highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            # Check for local high (resistance)
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                resistance_levels.append(high.iloc[i])
            
            # Check for local low (support)
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                support_levels.append(low.iloc[i])
        
        # Get most significant levels
        resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
        support_levels = sorted(set(support_levels))[:5]
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels
        }
    except Exception as e:
        logger.warning(f"Error calculating Support/Resistance: {e}")
        return None

def detect_market_structure_breaks(df, swing_strength=5):
    """تشخیص Market Structure Breaks"""
    try:
        if df is None or len(df) < swing_strength * 2:
            return None
        
        high = df['high']
        low = df['low']
        
        structure_breaks = pd.DataFrame(index=df.index)
        structure_breaks['bullish_break'] = False
        structure_breaks['bearish_break'] = False
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_strength, len(df) - swing_strength):
            # Swing High
            if high.iloc[i] == high.iloc[i-swing_strength:i+swing_strength+1].max():
                swing_highs.append((i, high.iloc[i]))
            
            # Swing Low
            if low.iloc[i] == low.iloc[i-swing_strength:i+swing_strength+1].min():
                swing_lows.append((i, low.iloc[i]))
        
        # Detect breaks
        current_price = df['close'].iloc[-1]
        
        # Check if current price breaks recent swing high (bullish break)
        if swing_highs:
            recent_high = max(swing_highs[-3:], key=lambda x: x[1])[1] if len(swing_highs) >= 3 else swing_highs[-1][1]
            if current_price > recent_high:
                structure_breaks.iloc[-1, structure_breaks.columns.get_loc('bullish_break')] = True
        
        # Check if current price breaks recent swing low (bearish break)
        if swing_lows:
            recent_low = min(swing_lows[-3:], key=lambda x: x[1])[1] if len(swing_lows) >= 3 else swing_lows[-1][1]
            if current_price < recent_low:
                structure_breaks.iloc[-1, structure_breaks.columns.get_loc('bearish_break')] = True
        
        return structure_breaks
    except Exception as e:
        logger.warning(f"Error detecting Market Structure Breaks: {e}")
        return None

# ===== فیلترهای اضافی =====

def calculate_correlation_with_btc(df, btc_df, period=20):
    """محاسبه همبستگی با بیت کوین"""
    try:
        if df is None or btc_df is None or len(df) < period or len(btc_df) < period:
            return None
            
        # هم‌تراز کردن داده‌ها بر اساس زمان
        merged = pd.merge(df[['close']], btc_df[['close']], 
                         left_index=True, right_index=True, 
                         suffixes=('', '_btc'), how='inner')
        
        if len(merged) < period:
            return None
            
        # محاسبه همبستگی غلتان
        correlation = merged['close'].rolling(period).corr(merged['close_btc'])
        
        return correlation
    except Exception:
        return None

def detect_market_regime(df, lookback=50):
    """تشخیص رژیم بازار"""
    try:
        if df is None or len(df) < lookback:
            return None
            
        close = df['close']
        
        # محاسبه نوسانات
        returns = close.pct_change()
        volatility = returns.rolling(lookback).std() * np.sqrt(252)  # سالانه
        
        # محاسبه ترند
        sma_short = close.rolling(10).mean()
        sma_long = close.rolling(50).mean()
        trend = sma_short - sma_long
        
        # تعیین رژیم بازار
        regime = pd.Series(index=df.index, dtype=str)
        
        for i in range(lookback, len(df)):
            vol = volatility.iloc[i]
            tr = trend.iloc[i]
            
            if vol > volatility.rolling(lookback).quantile(0.75).iloc[i]:
                if tr > 0:
                    regime.iloc[i] = 'Bull_Volatile'
                else:
                    regime.iloc[i] = 'Bear_Volatile'
            else:
                if tr > 0:
                    regime.iloc[i] = 'Bull_Stable'
                else:
                    regime.iloc[i] = 'Bear_Stable'
        
        return regime
    except Exception:
        return None

# ===== ابزارهای ریسک منجمنت =====

def calculate_position_size_atr(capital, risk_percent, entry_price, atr_value, atr_multiplier=2):
    """محاسبه اندازه پوزیشن بر اساس ATR"""
    try:
        risk_amount = capital * (risk_percent / 100)
        stop_distance = atr_value * atr_multiplier
        position_size = risk_amount / stop_distance
        
        return min(position_size, capital * 0.1)  # حداکثر 10% سرمایه
    except Exception:
        return 0

def calculate_dynamic_stop_loss(df, entry_price, position_type='long', atr_multiplier=2):
    """محاسبه حد ضرر پویا"""
    try:
        if df is None or len(df) < 14:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
            
        atr = calculate_atr(df)
        if atr is None:
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
            
        atr_value = atr.iloc[-1]
        
        if position_type == 'long':
            stop_loss = entry_price - (atr_value * atr_multiplier)
        else:
            stop_loss = entry_price + (atr_value * atr_multiplier)
            
        return stop_loss
    except Exception:
        return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

def optimize_risk_reward_ratio(entry_price, target_price, stop_loss, min_ratio=2.0):
    """بهینه‌سازی نسبت ریسک-ریوارد"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        current_ratio = reward / risk if risk > 0 else 0
        
        if current_ratio < min_ratio:
            # تنظیم هدف برای دستیابی به نسبت حداقل
            if entry_price > stop_loss:  # long position
                new_target = entry_price + (risk * min_ratio)
            else:  # short position
                new_target = entry_price - (risk * min_ratio)
            
            return new_target
        
        return target_price
    except Exception:
        return target_price

# ===== تکنیک‌های بهبود دقت =====

def ensemble_signal_scoring(signals_dict, weights=None):
    """ترکیب چندین سیگنال با وزن‌دهی"""
    try:
        if not signals_dict:
            return 0
            
        if weights is None:
            weights = {key: 1 for key in signals_dict.keys()}
        
        total_score = 0
        total_weight = 0
        
        for signal_name, signal_value in signals_dict.items():
            if signal_name in weights and signal_value is not None:
                weight = weights[signal_name]
                total_score += signal_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    except Exception:
        return 0

def adaptive_threshold_calculator(df, indicator_values, percentile_low=20, percentile_high=80):
    """محاسبه آستانه‌های تطبیقی"""
    try:
        if df is None or indicator_values is None:
            return {'low': 30, 'high': 70}
            
        # محاسبه آستانه‌ها بر اساس توزیع تاریخی
        low_threshold = np.percentile(indicator_values.dropna(), percentile_low)
        high_threshold = np.percentile(indicator_values.dropna(), percentile_high)
        
        return {
            'low': low_threshold,
            'high': high_threshold
        }
    except Exception:
        return {'low': 30, 'high': 70}
    
def calculate_market_microstructure(df, period=20):
    """محاسبه ساختار میکرو بازار"""
    try:
        if df is None or len(df) < period:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # Bid-Ask Spread Proxy
        spread_proxy = (high - low) / close
        avg_spread = spread_proxy.rolling(window=period).mean()
        
        # Market Depth Indicator
        price_impact = (high - low) / volume
        market_depth = price_impact.rolling(window=period).mean()
        
        # Order Flow Imbalance
        price_change = close.pct_change()
        volume_weighted_price_change = price_change * volume
        order_flow = volume_weighted_price_change.rolling(window=period).sum()
        
        # Liquidity Score
        liquidity_score = volume / (high - low)
        liquidity_score = liquidity_score.replace([np.inf, -np.inf], 0).fillna(0)
        avg_liquidity = liquidity_score.rolling(window=period).mean()
        
        return {
            'spread_proxy': avg_spread,
            'market_depth': market_depth,
            'order_flow': order_flow,
            'liquidity_score': avg_liquidity
        }
    except Exception as e:
        logger.warning(f"Error calculating market microstructure: {e}")
        return None

def calculate_support_resistance_levels(df, window=20, min_touches=3):
    """محاسبه سطوح حمایت و مقاومت دقیق"""
    try:
        if df is None or len(df) < window * 2:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # پیدا کردن نقاط pivot
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df) - window):
            # Pivot High
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                pivot_highs.append((i, high.iloc[i]))
            
            # Pivot Low
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                pivot_lows.append((i, low.iloc[i]))
        
        # تجمیع سطوح مشابه
        def cluster_levels(levels, tolerance=0.01):
            if not levels:
                return []
            
            levels = sorted(levels, key=lambda x: x[1])
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level[1] - current_cluster[-1][1]) / current_cluster[-1][1] <= tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [level]
            clusters.append(current_cluster)
            
            return clusters
        
        # تشکیل کلاسترهای حمایت و مقاومت
        resistance_clusters = cluster_levels(pivot_highs)
        support_clusters = cluster_levels(pivot_lows)
        
        # انتخاب قوی‌ترین سطوح
        strong_resistance = []
        for cluster in resistance_clusters:
            if len(cluster) >= min_touches:
                avg_price = sum(level[1] for level in cluster) / len(cluster)
                strength = len(cluster)
                strong_resistance.append((avg_price, strength))
        
        strong_support = []
        for cluster in support_clusters:
            if len(cluster) >= min_touches:
                avg_price = sum(level[1] for level in cluster) / len(cluster)
                strength = len(cluster)
                strong_support.append((avg_price, strength))
        
        # مرتب‌سازی بر اساس قدرت
        strong_resistance = sorted(strong_resistance, key=lambda x: x[1], reverse=True)[:5]
        strong_support = sorted(strong_support, key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'resistance_levels': [level[0] for level in strong_resistance],
            'support_levels': [level[0] for level in strong_support],
            'resistance_strength': [level[1] for level in strong_resistance],
            'support_strength': [level[1] for level in strong_support]
        }
    except Exception as e:
        logger.warning(f"Error calculating support/resistance levels: {e}")
        return None

def detect_dark_cloud_cover(df):
    """تشخیص الگوی Dark Cloud Cover"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.Series(False, index=df.index)
        
        for i in range(1, len(df)):
            # کندل اول: صعودی قوی
            first_bullish = close.iloc[i-1] > open_price.iloc[i-1]
            first_body = close.iloc[i-1] - open_price.iloc[i-1]
            
            # کندل دوم: نزولی
            second_bearish = close.iloc[i] < open_price.iloc[i]
            second_body = open_price.iloc[i] - close.iloc[i]
            
            # شرایط Dark Cloud Cover
            opens_above_prev_high = open_price.iloc[i] > high.iloc[i-1]  # باز شدن بالای کندل قبل
            closes_into_first_body = (close.iloc[i] < (open_price.iloc[i-1] + close.iloc[i-1]) / 2)  # بسته شدن در نیمه پایین کندل اول
            significant_penetration = second_body > first_body * 0.5  # نفوذ قابل توجه
            
            if (first_bullish and second_bearish and opens_above_prev_high and 
                closes_into_first_body and significant_penetration):
                patterns.iloc[i] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Dark Cloud Cover: {e}")
        return None

def detect_piercing_line(df):
    """تشخیص الگوی Piercing Line"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.Series(False, index=df.index)
        
        for i in range(1, len(df)):
            # کندل اول: نزولی قوی
            first_bearish = close.iloc[i-1] < open_price.iloc[i-1]
            first_body = open_price.iloc[i-1] - close.iloc[i-1]
            
            # کندل دوم: صعودی
            second_bullish = close.iloc[i] > open_price.iloc[i]
            second_body = close.iloc[i] - open_price.iloc[i]
            
            # شرایط Piercing Line
            opens_below_prev_low = open_price.iloc[i] < low.iloc[i-1]  # باز شدن زیر کندل قبل
            closes_into_first_body = (close.iloc[i] > (open_price.iloc[i-1] + close.iloc[i-1]) / 2)  # بسته شدن در نیمه بالای کندل اول
            significant_penetration = second_body > first_body * 0.5  # نفوذ قابل توجه
            
            if (first_bearish and second_bullish and opens_below_prev_low and 
                closes_into_first_body and significant_penetration):
                patterns.iloc[i] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Piercing Line: {e}")
        return None

def detect_harami_patterns(df):
    """تشخیص الگوهای Harami"""
    try:
        if df is None or len(df) < 2:
            return None
        
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        patterns = pd.DataFrame(index=df.index)
        patterns['bullish_harami'] = False
        patterns['bearish_harami'] = False
        
        for i in range(1, len(df)):
            # محاسبه اندازه بدنه کندل‌ها
            first_body_size = abs(close.iloc[i-1] - open_price.iloc[i-1])
            second_body_size = abs(close.iloc[i] - open_price.iloc[i])
            
            # کندل دوم باید در داخل کندل اول باشد
            first_max = max(open_price.iloc[i-1], close.iloc[i-1])
            first_min = min(open_price.iloc[i-1], close.iloc[i-1])
            second_max = max(open_price.iloc[i], close.iloc[i])
            second_min = min(open_price.iloc[i], close.iloc[i])
            
            is_inside = (second_max < first_max and second_min > first_min)
            is_smaller = second_body_size < first_body_size * 0.7  # کندل دوم کوچک‌تر
            
            # Bullish Harami
            first_bearish = close.iloc[i-1] < open_price.iloc[i-1]  # کندل اول نزولی
            second_bullish = close.iloc[i] > open_price.iloc[i]     # کندل دوم صعودی
            
            if first_bearish and second_bullish and is_inside and is_smaller:
                patterns.iloc[i, patterns.columns.get_loc('bullish_harami')] = True
            
            # Bearish Harami
            first_bullish = close.iloc[i-1] > open_price.iloc[i-1]  # کندل اول صعودی
            second_bearish = close.iloc[i] < open_price.iloc[i]     # کندل دوم نزولی
            
            if first_bullish and second_bearish and is_inside and is_smaller:
                patterns.iloc[i, patterns.columns.get_loc('bearish_harami')] = True
        
        return patterns
    except Exception as e:
        logger.warning(f"Error detecting Harami patterns: {e}")
        return None

def calculate_vwap(df):
    """محاسبه Volume Weighted Average Price"""
    try:
        if df is None or len(df) < 1:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # محاسبه Typical Price
        typical_price = (high + low + close) / 3
        
        # محاسبه VWAP
        cumulative_typical_price_volume = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        vwap = cumulative_typical_price_volume / cumulative_volume
        
        # محاسبه VWAP Bands (انحراف معیار)
        vwap_variance = ((typical_price - vwap) ** 2 * volume).cumsum() / cumulative_volume
        vwap_std = np.sqrt(vwap_variance)
        
        vwap_upper1 = vwap + vwap_std
        vwap_lower1 = vwap - vwap_std
        vwap_upper2 = vwap + 2 * vwap_std
        vwap_lower2 = vwap - 2 * vwap_std
        
        return {
            'vwap': vwap,
            'vwap_upper1': vwap_upper1,
            'vwap_lower1': vwap_lower1,
            'vwap_upper2': vwap_upper2,
            'vwap_lower2': vwap_lower2
        }
    except Exception as e:
        logger.warning(f"Error calculating VWAP: {e}")
        return None

def filter_false_signals(df, signal_data, min_volume_ratio=1.2, min_trend_strength=0.1):
    """فیلتر سیگنال‌های کاذب"""
    try:
        if df is None or not signal_data:
            return False
            
        last_row = df.iloc[-1]
        
        # فیلتر حجم
        if 'volume' in df.columns and len(df) >= 20:
            # Calculate volume SMA if not available
            volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
            if not pd.isna(volume_sma) and volume_sma > 0:
                volume_ratio = last_row['volume'] / volume_sma
                if volume_ratio < min_volume_ratio:
                    return False
        
        # فیلتر قدرت ترند
        if len(df) >= 10:
            recent_closes = df['close'].tail(10).values
            if len(recent_closes) > 0 and recent_closes[0] != 0:
                trend_strength = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
                
                # Check signal_data structure
                signal_type = None
                if isinstance(signal_data, dict) and 'type' in signal_data:
                    signal_type = signal_data['type']
                elif isinstance(signal_data, str):
                    signal_type = signal_data
                
                if signal_type == 'buy' and trend_strength < -min_trend_strength:
                    return False
                elif signal_type == 'sell' and trend_strength > min_trend_strength:
                    return False
        
        return True
    except Exception:
        return True

def calculate_market_structure_score(df, lookback=20):
    """Calculate market structure quality score"""
    try:
        if df is None or len(df) < lookback:
            return 0
        
        recent_data = df.tail(lookback)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        closes = recent_data['close'].values
        
        # Higher highs and higher lows for uptrend
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        
        # Lower highs and lower lows for downtrend
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        # Price momentum consistency
        up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        down_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
        
        # Volume trend consistency
        if 'volume' in recent_data.columns:
            volumes = recent_data['volume'].values
            volume_trend = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i-1])
            volume_consistency = volume_trend / (len(volumes) - 1) if len(volumes) > 1 else 0.5
        else:
            volume_consistency = 0.5
        
        # Calculate structure strength
        uptrend_strength = (higher_highs + higher_lows) / (2 * (lookback - 1))
        downtrend_strength = (lower_highs + lower_lows) / (2 * (lookback - 1))
        
        # Momentum consistency
        momentum_consistency = max(up_moves, down_moves) / (len(closes) - 1) if len(closes) > 1 else 0.5
        
        # Final structure score
        if uptrend_strength > downtrend_strength:
            structure_score = (uptrend_strength * 0.4 + momentum_consistency * 0.4 + volume_consistency * 0.2) * 100
        else:
            structure_score = (downtrend_strength * 0.4 + momentum_consistency * 0.4 + volume_consistency * 0.2) * 100
        
        return min(structure_score, 100)
        
    except Exception:
        return 0