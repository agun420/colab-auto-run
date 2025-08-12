# src/data_fetcher.py

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import warnings
from multiprocessing import Pool, TimeoutError

warnings.filterwarnings('ignore')

def get_sp500_symbols():
    """
    Scrape Wikipedia to get the list of S&P 500 stock symbols.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', {'id': 'constituents'})

        symbols = []
        for row in table.find_all('tr')[1:]:
            symbol = row.find('td').text.strip()
            symbols.append(symbol)

        print(f"Successfully fetched {len(symbols)} S&P 500 symbols.")
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']

def _fetch_stock_data_workaround(symbol_period):
    symbol, period = symbol_period
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    stock = yf.Ticker(symbol, session=session)
    hist = stock.history(period=period)
    info = stock.info
    return hist, info

def get_stock_data(symbol, period='1y'):
    """
    Fetch historical stock data with a timeout using multiprocessing.
    """
    with Pool(processes=1) as pool:
        try:
            result = pool.apply_async(_fetch_stock_data_workaround, ((symbol, period),))
            hist, info = result.get(timeout=30)
            if hist.empty:
                print(f"No historical data for {symbol}. Skipping.")
                return None, None
            return hist, info
        except TimeoutError:
            print(f"Timeout fetching stock data for {symbol}. Skipping.")
            return None, None
        except Exception as e:
            print(f"Error fetching data for {symbol} with yfinance: {e}")
            return None, None

def _fetch_financial_data_workaround(symbol):
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    stock = yf.Ticker(symbol, session=session)
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    info = stock.info
    return financials, balance_sheet, cash_flow, info

def get_financial_data(symbol):
    """
    Get fundamental financial data with a timeout using multiprocessing.
    """
    with Pool(processes=1) as pool:
        try:
            result = pool.apply_async(_fetch_financial_data_workaround, (symbol,))
            financials, balance_sheet, cash_flow, info = result.get(timeout=30)

            if financials.empty or balance_sheet.empty or cash_flow.empty:
                print(f"Incomplete financial data for {symbol}. Skipping.")
                return None

            return {
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'info': info
            }
        except TimeoutError:
            print(f"Timeout fetching financial data for {symbol}. Skipping.")
            return None
        except Exception as e:
            print(f"Error fetching financial statements for {symbol}: {e}")
            return None
