import os
import logging
import asyncio
import warnings
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pandas as pd
import numpy as np
import talib
import ccxt.async_support as ccxt
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'token')
exchange = None

def load_symbols():
    try:
        with open('symbols.txt','r',encoding='utf-8') as f:
            return [l.strip().upper() for l in f if l.strip()]
    except:
        default=['BTC/USDT','ETH/USDT','BNB/USDT','ADA/USDT','SOL/USDT','DOT/USDT','LINK/USDT','XRP/USDT','LTC/USDT','MATIC/USDT']
        with open('symbols.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(default))
        return default

def load_symbols():
    """Load symbols from file with error handling"""
    try:
        with open('symbols.txt', 'r', encoding='utf-8') as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from symbols.txt")
        return symbols
    except FileNotFoundError:
        logger.error("symbols.txt file not found. Using default symbols.")
        # Create default symbols.txt file
        default_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 
                            'DOT/USDT', 'LINK/USDT', 'XRP/USDT', 'LTC/USDT', 'MATIC/USDT']
        try:
            with open('symbols.txt', 'w', encoding='utf-8') as f:
                for symbol in default_symbols:
                    f.write(f"{symbol}\n")
            logger.info("Created default symbols.txt file")
        except Exception as e:
            logger.error(f"Could not create symbols.txt: {e}")
        return default_symbols
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

SYMBOLS = load_symbols()

async def init_exchange():
    global exchange
    if exchange is None:
        exchange = ccxt.coinex({'apiKey':os.getenv('COINEX_API_KEY',''),'secret':os.getenv('COINEX_SECRET',''),'enableRateLimit':True,'timeout':30000,'options':{'defaultType':'spot'}})
    return exchange

async def get_klines(symbol,interval='1h',limit=1500):
    for _ in range(3):
        try:
            ex = await init_exchange()
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            ohlcv = await asyncio.wait_for(ex.fetch_ohlcv(symbol, interval, limit=limit), timeout=20)
            if not ohlcv or len(ohlcv)<300:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            if len(df)<300:
                return None
            return df
        except:
            await asyncio.sleep(1)
    return None

async def get_current_price(symbol):
    for _ in range(2):
        try:
            ex = await init_exchange()
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            ticker = await asyncio.wait_for(ex.fetch_ticker(symbol), timeout=10)
            return float(ticker.get('last',0))
        except:
            await asyncio.sleep(1)
    return None

def calculate_indicators(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    df['sma20'] = talib.SMA(close,20)
    df['sma50'] = talib.SMA(close,50)
    df['sma200'] = talib.SMA(close,200)
    df['ema12'] = talib.EMA(close,12)
    df['ema26'] = talib.EMA(close,26)
    df['rsi'] = talib.RSI(close,14)
    macd, macdsignal, macdhist = talib.MACD(close,12,26,9)
    df['macd'] = macd; df['macdsignal'] = macdsignal; df['macdhist'] = macdhist
    upper, mid, lower = talib.BBANDS(close,20,2)
    df['bb_upper'] = upper; df['bb_mid'] = mid; df['bb_lower'] = lower
    k, d = talib.STOCH(high,low,close,14,3,3)
    df['stoch_k'] = k; df['stoch_d'] = d
    df['atr'] = talib.ATR(high, low, close, 14)
    df['adx'] = talib.ADX(high, low, close, 14)
    df['cci'] = talib.CCI(high, low, close, 20)
    df['obv'] = talib.OBV(close, volume)
    df['typical'] = (high+low+close)/3
    df['vwap'] = (df['typical']*df['volume']).cumsum()/df['volume'].cumsum()
    df['engulf'] = talib.CDLENGULFING(open=df['open'], high=df['high'], low=df['low'], close=close)
    return df.dropna()

def check_signals(df, symbol):
    if len(df)<2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    triggers = []
    if last['close']>last['bb_upper']:
        triggers.append('bb_breakout')
    if prev['sma50']<prev['close']<last['sma50']<last['close']:
        triggers.append('sma50_cross')
    if last['adx']>25:
        triggers.append('strong_trend')
    if last['obv']>df['obv'].rolling(50).mean().iloc[-1]:
        triggers.append('obv_support')
    if last['engulf']!=0:
        triggers.append('engulfing')
    rsi = last['rsi']
    macd = last['macd']; sig = last['macdsignal']
    vol = last['volume']; vs = last['vol_sma']
    buy = prev['macd']<=prev['macdsignal'] and macd>sig and vol>vs*1.2 and rsi<40
    sell = prev['macd']>=prev['macdsignal'] and macd<sig and vol>vs*1.2 and rsi>60
    if buy:
        return {'type':'buy','rsi':rsi,'macd':macd,'atr':last['atr'],'vwap':last['vwap'],'triggers':triggers}
    if sell:
        return {'type':'sell','rsi':rsi,'macd':macd,'atr':last['atr'],'vwap':last['vwap'],'triggers':triggers}
    if rsi<25:
        return {'type':'buy','rsi':rsi,'macd':macd,'atr':last['atr'],'vwap':last['vwap'],'triggers':triggers}
    if rsi>75:
        return {'type':'sell','rsi':rsi,'macd':macd,'atr':last['atr'],'vwap':last['vwap'],'triggers':triggers}
    return None

def calculate_strength(df, sd):
    score = 0
    rsi = sd['rsi']
    vol_ratio = df.iloc[-1]['volume']/df.iloc[-1]['vol_sma']
    score += 3 if (sd['type']=='buy' and rsi<20) or (sd['type']=='sell' and rsi>80) else 2 if (sd['type']=='buy' and rsi<25) or (sd['type']=='sell' and rsi>75) else 1 if (sd['type']=='buy' and rsi<30) or (sd['type']=='sell' and rsi>70) else 0
    score += 2 if vol_ratio>2 else 1 if vol_ratio>1.5 else 0
    score += 1 if abs(sd['macd'])>0.001 else 0
    score += 1 if sd['atr']>df['atr'].rolling(50).mean().iloc[-1] else 0
    score += 1 if 'bb_breakout' in sd.get('triggers',[]) else 0
    score += 1 if 'engulfing' in sd.get('triggers',[]) else 0
    return min(max(score,1),8)

def calculate_accuracy(df, sd, symbol):
    score = 0
    last = df.iloc[-1]
    rsi = sd['rsi']
    score += 25 if (sd['type']=='buy' and rsi<20) or (sd['type']=='sell' and rsi>80) else 20 if (sd['type']=='buy' and rsi<25) or (sd['type']=='sell' and rsi>75) else 15 if (sd['type']=='buy' and rsi<30) or (sd['type']=='sell' and rsi>70) else 10 if (sd['type']=='buy' and rsi<35) or (sd['type']=='sell' and rsi>65) else 0
    macdh = last['macd']-last['macdsignal']
    score += 20 if (sd['type']=='buy' and macdh>0) or (sd['type']=='sell' and macdh<0) else 10 if abs(macdh)>0.001 else 0
    volr = last['volume']/last['vol_sma']
    score += 15 if volr>3 else 12 if volr>2 else 8 if volr>1.5 else 5 if volr>1.2 else 0
    cp, s20, s50, s200 = last['close'], last['sma20'], last['sma50'], last['sma200']
    if sd['type']=='buy':
        score += 15 if cp>s20>s50>s200 else 10 if cp>s20>s50 else 5 if cp>s20 else 0
    else:
        score += 15 if cp<s20<s50<s200 else 10 if cp<s20<s50 else 5 if cp<s20 else 0
    k, d = last['stoch_k'], last['stoch_d']
    score += 10 if (sd['type']=='buy' and k<20 and d<20) or (sd['type']=='sell' and k>80 and d>80) else 5 if (sd['type']=='buy' and k<30) or (sd['type']=='sell' and k>70) else 0
    score += 10 if last['adx']>30 else 5 if last['adx']>25 else 0
    prev = df['close'].iloc[-10:].values
    trend = sum(1 if prev[i]>prev[i-1] else -1 for i in range(1,len(prev)))
    score += int(10*(abs(trend)/len(prev))) if (sd['type']=='buy' and trend>0) or (sd['type']=='sell' and trend<0) else 0
    score += 5 if symbol in ['BTC/USDT','ETH/USDT','BNB/USDT'] else 0
    return min(score,100)

def ml_predict(df):
    features = ['rsi','macd','atr','adx','vwap']
    data = df.dropna()[features]
    target = (df['close'].shift(-1)>df['close']).astype(int).dropna()
    if len(data)>50:
        model = LogisticRegression(solver='liblinear')
        model.fit(data[:-1], target[:-1])
        return model.predict_proba(data.iloc[-1:].values)[0][1]
    return 0.5

async def analyze_market():
    signals = []
    for sym in SYMBOLS:
        await asyncio.sleep(1)
        df = await get_klines(sym)
        if df is None:
            continue
        df = calculate_indicators(df)
        sd = check_signals(df, sym)
        if sd:
            price = await get_current_price(sym)
            if not price:
                continue
            ml_score = ml_predict(df)
            if ml_score<0.6:
                continue
            acc = calculate_accuracy(df, sd, sym)
            if acc<50:
                continue
            strg = calculate_strength(df, sd)
            entry = price
            if sd['type']=='buy':
                target = entry*(1+0.05+sd['atr']/entry)
                sl = entry*(1-0.03)
                stype = 'Long'
            else:
                target = entry*(1-0.05-sd['atr']/entry)
                sl = entry*(1+0.03)
                stype = 'Short'
            signals.append({
                'symbol': sym,
                'type': stype,
                'entry': entry,
                'target': target,
                'stop_loss': sl,
                'strength': strg,
                'accuracy': acc,
                'ml_score': ml_score,
                'rsi': sd['rsi'],
                'macd': sd['macd'],
                'atr': sd['atr'],
                'vwap': sd['vwap'],
                'triggers': sd.get('triggers',[]),
                'time': datetime.now().strftime('%H:%M:%S')
            })
    return sorted(signals, key=lambda x: x['accuracy']*x['ml_score'], reverse=True)[:1]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 تحلیل بازار در حال انجام...")
    sigs = await asyncio.wait_for(analyze_market(), timeout=300)
    if sigs:
        s = sigs[0]
        emoji = '📈' if s['type']=='Long' else '📉'
        profit = ((s['target']-s['entry'])/s['entry']*100) if s['type']=='Long' else ((s['entry']-s['target'])/s['entry']*100)
        loss = ((s['entry']-s['stop_loss'])/s['entry']*100) if s['type']=='Long' else ((s['stop_loss']-s['entry'])/s['entry']*100)
        msg = f"{emoji} *{s['type']} {s['symbol']}*\nامتیاز: {s['accuracy']}/100\nML: {s['ml_score']:.2f}\nورودی: `{s['entry']:.6f}`\nهدف: `{s['target']:.6f}` (+{profit:.1f}%)\nحد ضرر: `{s['stop_loss']:.6f}` (-{loss:.1f}%)\nRSI: `{s['rsi']:.1f}`\nMACD: `{s['macd']:.6f}`\nATR: `{s['atr']:.6f}`\nVWAP: `{s['vwap']:.6f}`\nروش‌ها: `{','.join(s['triggers'])}`\nقدرت: {'⭐'*s['strength']}\n⏰ `{s['time']}`"
    else:
        msg = "❌ سیگنال با دقت بالا یافت نشد."
    await update.message.reply_text(msg, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ex = await init_exchange()
    stat = "✅" if await ex.fetch_ticker('BTC/USDT') else "❌"
    msg = f"🤖 ربات فعال\nنمادها: {len(SYMBOLS)}\nCoinEx: {stat}\n⏰ `{datetime.now().strftime('%H:%M:%S')}`"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def show_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = "\n".join(" | ".join(SYMBOLS[i:i+3]) for i in range(0,len(SYMBOLS),3))
    await update.message.reply_text(f"📋 نمادها:\n{txt}", parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📋 دستورات:\n/start\n/status\n/symbols\n/help", parse_mode='Markdown')

async def close_exchange():
    global exchange
    if exchange:
        await exchange.close()
        exchange = None

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("symbols", show_symbols))
    app.add_handler(CommandHandler("help", help_command))
    app.run_polling()

if __name__ == "__main__":
    main()
