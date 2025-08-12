# src/reporting.py

import requests
from datetime import datetime
from src import config

def send_telegram_message(message):
    """
    Send a formatted message to the configured Telegram chat.
    """
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': config.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Telegram API error: {response.status_code} - {response.text}")
            return False
        return True
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        return False

def generate_report(stocks):
    """
    Generate a formatted report from the list of recommended stocks.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    report = f"📊 <b>Daily Stock Screen Report - {today}</b>\n"

    if not stocks:
        report += "\nNo stocks passed all screening criteria today. Market conditions may not be favorable."
        return report

    report += f"Found {len(stocks)} high-probability opportunities:\n\n"

    for i, stock in enumerate(stocks, 1):
        symbol = stock['symbol']
        price = stock['current_price']
        market_cap = stock.get('market_cap', 0) / 1e9  # Convert to billions
        model_score = stock.get('model_prediction_score', 0) * 100

        report += f"<b>{i}. {symbol}</b> - ${price:.2f} (${market_cap:.1f}B)\n"
        report += f"✨ <b>Prediction Score: {model_score:.1f}%</b>\n"

        # Key signals
        report += "🎯 <b>Key Signals:</b>\n"
        for signal in stock.get('tier1_signals', [])[:3]:
            report += f"  • {signal}\n"

        # Targets
        targets = stock.get('targets', {})
        if targets:
            report += f"📈 <b>Targets:</b>\n"
            report += f"  • Swing (1-5d): ${targets.get('st_target1', 0):.2f} / ${targets.get('st_target2', 0):.2f}\n"
            report += f"  • Position (6-18m): ${targets.get('target1', 0):.2f} / ${targets.get('target2', 0):.2f}\n"
            report += f"  • Stop Loss: ${targets.get('stop_loss', 0):.2f}\n"

        report += "\n"

    report += "⚠️ <b>Disclaimer:</b> This is an automated analysis. Always conduct your own research."
    return report

def send_startup_message():
    """
    Send a notification when the screener starts.
    """
    startup_message = f"🤖 Stock Screener activated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if send_telegram_message(startup_message):
        print("✅ Telegram connection successful.")
        return True
    else:
        print("❌ Telegram connection failed. Please check your token and chat ID.")
        return False
