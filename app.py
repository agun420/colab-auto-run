# app.py

import pandas as pd
from src.data_fetcher import get_sp500_symbols
from src.screener import StockScreener
from src.reporting import generate_report, send_telegram_message, send_startup_message
from src import config

def main():
    """
    Main function to run the stock screening application.
    """
    if not send_startup_message():
        return

    # 1. Get the stock universe
    # Use the hardcoded list from config if it exists, otherwise fetch S&P 500
    stock_universe = getattr(config, 'STOCK_UNIVERSE', None) or get_sp500_symbols()

    if not stock_universe:
        print("Could not retrieve stock universe. Exiting.")
        return

    # 2. Initialize the screener
    screener = StockScreener()

    # 3. Screen all stocks
    passed_stocks = []
    for symbol in stock_universe:
        try:
            result = screener.screen_stock(symbol)
            if result:
                passed_stocks.append(result)
        except Exception as e:
            print(f"An unexpected error occurred while screening {symbol}: {e}")

    # 4. Sort results and generate the report
    if passed_stocks:
        # Sort by the model's prediction score in descending order
        passed_stocks.sort(key=lambda x: x['model_prediction_score'], reverse=True)

        # Get the top 5 recommendations
        top_5_stocks = passed_stocks[:5]

        report = generate_report(top_5_stocks)
    else:
        report = generate_report([]) # Generate the "no stocks found" report

    # 5. Send the final report
    print("\nSending final report to Telegram...")
    send_telegram_message(report)
    print("Screening complete. Report sent.")

if __name__ == "__main__":
    main()
