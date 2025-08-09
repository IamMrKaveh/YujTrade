Trading Signal Bot
This project is a trading bot that utilizes technical analysis to generate buy and sell signals for cryptocurrencies and sends them to users via Telegram. It leverages various technical indicators such as Moving Averages, RSI, MACD, and more to analyze market conditions and trends.
Features

Generates buy and sell signals based on technical analysis
Supports multiple technical indicators (e.g., SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, etc.)
Analyzes market conditions and trends
Dynamically calculates entry, exit, and stop-loss levels
Ranks signals based on various criteria
Integrates with the CoinEx exchange for market data
Delivers signals and analysis to users through Telegram
Includes backtesting and paper trading simulation capabilities

Installation and Setup

Install Dependencies:
pip install -r requirements.txt


Set Environment Variables:

COINEX_API_KEY: Your CoinEx API key
COINEX_SECRET: Your CoinEx API secret
BOT_TOKEN: Your Telegram bot token

You can set these in your environment or in a .env file:
export COINEX_API_KEY='your_api_key'
export COINEX_SECRET='your_api_secret'
export BOT_TOKEN='your_bot_token'


Run the Bot:
python main-gpt.py



Usage
Once the bot is running, interact with it via Telegram using these commands:

/start: Initiates the bot and displays the main menu
/config: Shows the current bot configuration
/quick: Performs a quick analysis on the 1-minute timeframe

You can also use the inline buttons in the main menu to request a full analysis or quick scan.
Configuration
The bot’s settings are stored in config.json. Edit this file to customize the bot’s behavior. Default settings include:

symbols: List of trading symbols (e.g., BTC/USDT)
timeframes: List of timeframes to analyze (e.g., 1m, 5m, 1h)
min_confidence_score: Minimum confidence score for signals (default: 50)
max_signals_per_timeframe: Maximum signals per timeframe (default: 3)
risk_reward_threshold: Minimum risk/reward ratio (default: 1.5)

Example config.json:
{
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "timeframes": ["1m", "5m", "15m"],
  "min_confidence_score": 60,
  "max_signals_per_timeframe": 3,
  "risk_reward_threshold": 2.0
}

Code Structure
The code is modular and consists of the following key components:

Technical Indicators: Classes like MovingAverageIndicator, RSIIndicator, MACDIndicator, etc., for calculating various indicators
Market Analysis: Classes such as MarketConditionAnalyzer, TrendAnalyzer, and SupportResistanceAnalyzer for market evaluation
Dynamic Levels: DynamicLevelCalculator for computing entry, exit, and stop-loss levels
Signal Generation: SignalGenerator for producing trading signals
Exchange Management: ExchangeManager for interacting with CoinEx and fetching OHLCV data
Signal Ranking: SignalRanking for prioritizing signals
Configuration: ConfigManager for handling bot settings
Bot Service: TradingBotService for managing analysis and signals
Backtesting & Simulation: BacktestingEngine and PaperTradingSimulator for testing strategies
Message Formatting: MessageFormatter for crafting Telegram messages
Telegram Handler: TelegramBotHandler for user interaction via Telegram

Dependencies

pandas: Data manipulation and analysis
ccxt: Cryptocurrency exchange trading library
numpy: Numerical computations
python-telegram-bot: Telegram bot API integration
sqlite3: Local database for caching OHLCV data

Install them using:
pip install pandas ccxt numpy python-telegram-bot sqlite3

License
This project is licensed under the GPL V3 License.