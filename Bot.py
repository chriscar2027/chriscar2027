#!/usr/bin/env python3
"""
Plug-and-Play trading decision engine (refactor of v1.002).
Usage:
    python bot.py --ticker NVDA
    python bot.py --ticker NVDA --headline "Nvidia announces massive $50B buyback"
Functions are exported for testing.
"""
import argparse
import logging
import sys
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Ensure NLTK VADER is available
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    logger.info("Downloading VADER lexicon...")
    nltk.download("vader_lexicon", quiet=True)

vader = SentimentIntensityAnalyzer()


# ---------- Helper utilities ----------
def safe_download(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download data via yfinance with safe fallbacks.
    Returns an empty DataFrame if download failed.
    """
    try:
        df = yf.download(ticker, period=period, progress=False, multi_level_index=False)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.warning(f"yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()


# ---------- Feature engineering ----------
def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["ADX"] = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"]
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["Volatility"] = df["Close"].pct_change().rolling(window=20).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df.dropna(inplace=True)
    return df


# ---------- Scout (signal logic) ----------
def scout_strategy(row: pd.Series) -> int:
    if (row["ADX"] > 25 and row["Close"] > row["SMA_50"]) or (row["ADX"] < 20 and row["RSI"] < 30):
        return 1
    return 0


# ---------- Model training (General) ----------
def train_model(data: pd.DataFrame):
    df = data.copy()
    df["Future_Return"] = df["Close"].shift(-5)
    df["Target_Label"] = (df["Future_Return"] > df["Close"] * 1.02).astype(int)
    df["Scout_Signal"] = df.apply(scout_strategy, axis=1)
    training_data = df[df["Scout_Signal"] == 1].copy()
    training_data.dropna(inplace=True)
    features = ["ADX", "RSI", "Volatility", "Volume_Change", "SMA_50"]
    if training_data.empty or len(training_data) < 5:
        # Not enough data to train; return None
        return None, features
    X = training_data[features]
    y = training_data["Target_Label"]
    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    model.fit(X, y)
    # Optional performance metric
    if len(X) > 50:
        split = int(len(X) * 0.8)
        X_test = X.iloc[split:]
        y_test = y.iloc[split:]
        prec = precision_score(y_test, model.predict(X_test), zero_division=0)
        logger.info(f"General's historical precision on buy signals: {prec:.2%}")
    return model, features


# ---------- Shield (sentiment veto) ----------
def get_sentiment_veto(headline_text: str) -> bool:
    score = vader.polarity_scores(headline_text)["compound"]
    logger.info(f"NEWS SCAN: '{headline_text}' -> SENTIMENT SCORE: {score:.2f}")
    return score < -0.5


# ---------- Map (macro risk sizing) ----------
def get_macro_risk() -> float:
    logger.info("SCANNING MACRO ECONOMY...")
    multiplier = 1.0

    def get_latest_change(ticker_symbol: str) -> Tuple[float, bool]:
        df = safe_download(ticker_symbol, period="5d")
        if df.empty or "Close" not in df.columns or df["Close"].isnull().all():
            return 0.0, False
        return df["Close"].pct_change().iloc[-1], True

    tnx_change, tnx_valid = get_latest_change("^TNX")
    if tnx_valid and tnx_change > 0.03:
        logger.info(f"[WARNING] Yields Spiked {tnx_change:.2%}. -50% size.")
        multiplier -= 0.5

    btc_change, btc_valid = get_latest_change("BTC-USD")
    if btc_valid and btc_change < -0.05:
        logger.info(f"[WARNING] Bitcoin Crashed {btc_change:.2%}. -30% size.")
        multiplier -= 0.3

    final_multiplier = max(0.0, multiplier)
    logger.info(f"CALCULATED RISK MULTIPLIER: {final_multiplier:.2f}")
    return final_multiplier


# ---------- Live decision ----------
def decide_last_day(data: pd.DataFrame, model, features, latest_headline: str = "") -> Dict:
    """
    Make final decision for the latest day. Returns dict with decision details.
    """
    out = {
        "scout_signal": 0,
        "model_confidence": None,
        "vetoed_by_sentiment": False,
        "final_size": 0,
        "reason": "No signal or insufficient data",
    }
    if data.empty:
        out["reason"] = "No data"
        return out

    latest_data = data.iloc[-1]
    out["scout_signal"] = scout_strategy(latest_data)
    if out["scout_signal"] != 1:
        out["reason"] = "No technical signal"
        return out

    if model is None:
        out["reason"] = "No trained model (not enough historical signal data)"
        return out

    current_features = latest_data[features].values.reshape(1, -1)
    try:
        confidence = model.predict_proba(current_features)[0][1]
    except Exception:
        confidence = float(model.predict(current_features)[0])  # fallback
    out["model_confidence"] = float(confidence)

    dynamic_threshold = min(0.55 + (latest_data["Volatility"] * 5.0), 0.90)
    if confidence <= dynamic_threshold:
        out["reason"] = f"Confidence too low ({confidence:.2%} <= {dynamic_threshold:.2%})"
        return out

    # Sentiment veto
    if latest_headline and get_sentiment_veto(latest_headline):
        out["vetoed_by_sentiment"] = True
        out["reason"] = "Vetoed by sentiment"
        return out

    # Position sizing
    risk_scaler = get_macro_risk()
    standard_size = 100
    final_size = int(standard_size * risk_scaler)
    if final_size <= 0:
        out["reason"] = "Blocked by macro map (size 0)"
        out["final_size"] = 0
        return out

    out["final_size"] = final_size
    out["reason"] = "BUY"
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Trading Decision Engine")
    parser.add_argument("--ticker", required=True, help="Ticker to analyze (e.g., NVDA)")
    parser.add_argument("--period", default="2y", help="yfinance period (default: 2y)")
    parser.add_argument("--headline", default="Nvidia announces massive $50 billion share buyback",
                        help="Headline text to simulate sentiment veto")
    args = parser.parse_args(argv)

    logger.info("--- SYSTEM STARTUP 🚀 ---")
    logger.info(f"1. Downloading data for {args.ticker}...")
    data = safe_download(args.ticker, period=args.period)
    if data.empty:
        logger.error(f"Error: No data found for {args.ticker}. Exiting.")
        sys.exit(1)
    logger.info(f"   -> Data fetched successfully up to {data.index[-1].strftime('%Y-%m-%d')}")

    data = compute_indicators(data)
    model, features = train_model(data)

    decision = decide_last_day(data, model, features, latest_headline=args.headline)
    logger.info("\n==================================")
    logger.info("   FINAL TRADING DECISION ENGINE   ")
    logger.info("==================================")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Decision: {decision['reason']}")
    if decision["final_size"]:
        logger.info(f"Quantity: {decision['final_size']} shares")
    if decision["model_confidence"] is not None:
        logger.info(f"Model Confidence: {decision['model_confidence']:.2%}")
    if decision["vetoed_by_sentiment"]:
        logger.info("Vetoed by sentiment shield.")
    return decision


if __name__ == "__main__":
    main()
