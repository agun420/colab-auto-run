# src/config.py

"""
Configuration settings for the stock screener.

Replace the placeholder values with your actual API keys and settings.
"""

# Telegram Bot Configuration
# Get your token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN = "7604672791:AAFauy6Nakx1hTMgbdbGFuqRhqEtPHcMyiw"
# Get your chat ID from @userinfobot on Telegram
TELEGRAM_CHAT_ID = "6970413519"

# API Keys for Financial Data
# Get your free API key from https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_KEY = "8AYVH3X3BTUT20CR"
# Get your free API key from https://finnhub.io/
FINNHUB_KEY = "d1dhtnpr01qn1ojnonggd1dhtnpr01qn1ojnonh0"

# Stock Universe
# By default, we will use the S&P 500 index.
# You can override this with a smaller list for testing.
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'CRM', 'NFLX'
]

# Screening Parameters
# You can adjust these thresholds to fit your investment strategy.
SCREENING_PARAMETERS = {
    'min_revenue_growth': 15.0,
    'min_margin_expansion': True,
    'require_positive_fcf': True,
    'max_peg_ratio': 1.5,
    'min_rsi': 35.0,
    'max_rsi': 70.0,
    'min_volume_ratio': 1.5,
    'min_5-day_momentum': 5.0
}

# Predictive Model Settings
MODEL_TRAINING_YEARS = 5
PREDICTION_HORIZON_DAYS = 21  # Predict performance over the next month
MODEL_THRESHOLD = 0.65  # Minimum probability to consider a stock a "buy"
