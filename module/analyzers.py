# analyzers.py

class VolumeAnalyzer:
    def analyze_volume_pattern(self, data: pd.DataFrame) -> Dict[str, float]:
        if len(data) < 20 or 'volume' not in data.columns or data['volume'].isnull().all():
            return {}
        volume_ma_20 = data["volume"].rolling(20).mean().iloc[-1]
        if pd.isna(volume_ma_20) or volume_ma_20 == 0:
            return {}
        current_volume = data["volume"].iloc[-1]
        ratio = current_volume / volume_ma_20
        return {"volume_ratio": ratio}