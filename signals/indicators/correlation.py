import pandas as pd

def _calculate_correlation_with_btc(df, btc_df, period=20):
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
