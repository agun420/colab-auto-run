# src/screener.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from src import config
from src.data_fetcher import get_stock_data, get_financial_data

class StockScreener:
    def __init__(self):
        self.params = config.SCREENING_PARAMETERS
        self.model = self.load_or_train_model()

    def _calculate_revenue_growth(self, financials):
        try:
            if financials is None or financials.empty: return 0
            revenue_row = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else financials.loc['Revenues']
            if len(revenue_row) < 2: return 0
            growth = ((revenue_row.iloc[0] - revenue_row.iloc[1]) / abs(revenue_row.iloc[1])) * 100
            return growth
        except:
            return 0

    def _calculate_margin_trend(self, financials):
        try:
            if financials is None or financials.empty: return False
            margin_row = financials.loc['Gross Profit'] if 'Gross Profit' in financials.index else financials.loc['Operating Income']
            if len(margin_row) < 2: return False
            return margin_row.iloc[0] > margin_row.iloc[1]
        except:
            return False

    def _check_fcf_positive(self, cash_flow):
        try:
            if cash_flow is None or cash_flow.empty: return False
            fcf_row = cash_flow.loc['Free Cash Flow']
            return fcf_row.iloc[0] > 0
        except:
            return False

    def get_fundamental_signals(self, symbol):
        data = get_financial_data(symbol)
        if not data: return {}

        financials = data['financials']
        info = data['info']

        return {
            'revenue_growth': self._calculate_revenue_growth(financials),
            'margin_trend': self._calculate_margin_trend(financials),
            'fcf_positive': self._check_fcf_positive(data['cash_flow']),
            'debt_equity': info.get('debtToEquity'),
            'forward_pe': info.get('forwardPE'),
            'trailing_pe': info.get('trailingPE'),
            'peg_ratio': info.get('pegRatio'),
            'market_cap': info.get('marketCap'),
        }

    def get_technical_signals(self, hist_data):
        try:
            if hist_data is None or hist_data.empty: return {}

            hist_data['MA_20'] = hist_data['Close'].rolling(20).mean()
            hist_data['MA_50'] = hist_data['Close'].rolling(50).mean()

            delta = hist_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            avg_volume = hist_data['Volume'].rolling(20).mean()
            volume_ratio = hist_data['Volume'].iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1

            current_price = hist_data['Close'].iloc[-1]

            return {
                'current_price': current_price,
                'above_ma20': current_price > hist_data['MA_20'].iloc[-1],
                'above_ma50': current_price > hist_data['MA_50'].iloc[-1],
                'rsi': rsi.iloc[-1],
                'volume_ratio': volume_ratio,
                'price_change_5d': ((current_price - hist_data['Close'].iloc[-6]) / hist_data['Close'].iloc[-6]) * 100
            }
        except:
            return {}

    def run_rule_based_screening(self, fundamentals, technicals):
        signals, catalysts, checks = [], [], []

        # Tier 1: Financial Performance
        if fundamentals.get('revenue_growth', 0) > self.params['min_revenue_growth']:
            signals.append(f"Revenue growth > {self.params['min_revenue_growth']}%")
        if fundamentals.get('margin_trend') == self.params['min_margin_expansion']:
            signals.append("Expanding margins")
        if fundamentals.get('fcf_positive') == self.params['require_positive_fcf']:
            signals.append("Positive FCF")

        # Tier 2: Catalysts
        if technicals.get('price_change_5d', 0) > self.params['min_5-day_momentum']:
            catalysts.append(f"5-day momentum > {self.params['min_5-day_momentum']}%")
        if technicals.get('volume_ratio', 1) > self.params['min_volume_ratio']:
            catalysts.append(f"High volume (>{self.params['min_volume_ratio']}x avg)")

        # Tier 3: Valuation & Health
        if 0 < fundamentals.get('peg_ratio', 999) < self.params['max_peg_ratio']:
            checks.append(f"PEG < {self.params['max_peg_ratio']}")
        if self.params['min_rsi'] <= technicals.get('rsi', 50) <= self.params['max_rsi']:
            checks.append("Healthy RSI")

        passed = len(signals) >= 2 and len(catalysts) >= 1 and len(checks) >= 1
        return passed, signals, catalysts, checks

    def calculate_targets(self, current_price):
        return {
            'st_target1': current_price * 1.05,
            'st_target2': current_price * 1.10,
            'target1': current_price * 1.25,
            'target2': current_price * 1.50,
            'stop_loss': current_price * 0.85,
        }

    def engineer_features(self, hist, fundamentals):
        """
        Create a feature set from historical data and fundamental signals.
        """
        features = pd.DataFrame()

        # Price and volume features
        features['returns'] = hist['Close'].pct_change()
        features['MA_20_diff'] = hist['MA_20'] - hist['Close']
        features['MA_50_diff'] = hist['MA_50'] - hist['Close']
        features['volume_change'] = hist['Volume'].pct_change()

        # Lag features
        for i in range(1, 6):
            features[f'returns_lag_{i}'] = features['returns'].shift(i)

        # Fundamental features (broadcasted across all rows)
        features['forward_pe'] = fundamentals.get('forward_pe', 0)
        features['peg_ratio'] = fundamentals.get('peg_ratio', 0)
        features['revenue_growth'] = fundamentals.get('revenue_growth', 0)

        # Target variable: Did the stock price increase significantly in the next N days?
        horizon = config.PREDICTION_HORIZON_DAYS
        future_price = hist['Close'].shift(-horizon)
        features['target'] = (future_price > hist['Close'] * 1.05).astype(int)

        return features.dropna()

    def load_or_train_model(self):
        """
        Train a Gradient Boosting model on a sample stock (e.g., SPY)
        to make the screener predictive. In a real-world scenario, this
        model would be trained on a much larger dataset and saved to a file.
        """
        print("Training predictive model on SPY data as a proxy for market behavior...")
        try:
            # We'll train a simple model on SPY data as a proxy for the market
            spy_hist, _ = get_stock_data('SPY', period=f"{config.MODEL_TRAINING_YEARS}y")
            spy_fundamentals = self.get_fundamental_signals('SPY')

            if spy_hist is None or not spy_fundamentals:
                print("Could not fetch SPY data for model training. Using a dummy model.")
                return None # Return a dummy model if data is unavailable

            features = self.engineer_features(spy_hist, spy_fundamentals)

            if len(features) < 50: # Need enough data to train
                print("Not enough data for model training. Using a dummy model.")
                return None

            X = features.drop('target', axis=1)
            y = features['target']

            # Split data without shuffling to maintain time series order
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X_train, y_train)

            print("✅ Predictive model trained successfully.")
            return model
        except Exception as e:
            print(f"Error during model training: {e}. Falling back to a dummy model.")
            return None # Return a dummy model in case of any error

    def screen_stock(self, symbol):
        try:
            start_time = datetime.now()
            print(f"Screening {symbol}...")

            # --- Data Fetching ---
            fetch_start = datetime.now()
            hist_data, info = get_stock_data(symbol, period=f"{config.MODEL_TRAINING_YEARS}y")
            if hist_data is None or hist_data.empty:
                print(f"No data for {symbol}, skipping.")
                return None
            print(f"[{symbol}] Data fetching took: {datetime.now() - fetch_start}")

            # --- Signal Calculation ---
            signal_start = datetime.now()
            fundamentals = self.get_fundamental_signals(symbol)
            technicals = self.get_technical_signals(hist_data)
            if not fundamentals or not technicals:
                print(f"[{symbol}] Could not calculate signals, skipping.")
                return None
            print(f"[{symbol}] Signal calculation took: {datetime.now() - signal_start}")

            # --- Rule-Based Screening ---
            rules_start = datetime.now()
            passed_rules, signals, _, _ = self.run_rule_based_screening(fundamentals, technicals)
            if not passed_rules:
                print(f"❌ {symbol} failed rule-based screening.")
                return None
            print(f"[{symbol}] Rule-based screening took: {datetime.now() - rules_start}")

            # --- Predictive Model Screening ---
            model_start = datetime.now()
            prediction_score = 0
            if self.model:
                features = self.engineer_features(hist_data, fundamentals)
                if not features.empty:
                    latest_features = features.drop('target', axis=1).iloc[-1:]
                    prediction_score = self.model.predict_proba(latest_features)[0][1]

            if prediction_score < config.MODEL_THRESHOLD:
                print(f"❌ {symbol} failed predictive model (Score: {prediction_score:.2f})")
                return None
            print(f"[{symbol}] Model screening took: {datetime.now() - model_start}")

            # --- Success ---
            total_time = datetime.now() - start_time
            print(f"✅ {symbol} passed all criteria in {total_time.total_seconds():.2f}s (Score: {prediction_score:.2f})")

            return {
                'symbol': symbol,
                'current_price': technicals['current_price'],
                'market_cap': fundamentals.get('market_cap'),
                'tier1_signals': signals,
                'targets': self.calculate_targets(technicals['current_price']),
                'model_prediction_score': prediction_score
            }
        except Exception as e:
            print(f"An unexpected error occurred while screening {symbol}: {e}")
            return None
