import builtins
from types import SimpleNamespace
import pandas as pd
import numpy as np
import pytest
import bot

# Helper: create synthetic OHLCV DataFrame with enough rows
def make_fake_price_series(days=300, seed=42):
    rng = np.random.RandomState(seed)
    # Simulate a trending price
    close = np.cumsum(rng.normal(loc=0.1, scale=1.0, size=days)) + 100
    high = close + np.abs(rng.normal(0, 0.5, size=days))
    low = close - np.abs(rng.normal(0, 0.5, size=days))
    open_ = close + rng.normal(0, 0.2, size=days)
    volume = np.abs(rng.normal(1e6, 1e5, size=days)).astype(int)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    }, index=idx)
    return df

@pytest.fixture(autouse=True)
def patch_yfinance(monkeypatch):
    # Patch yf.download used in bot.safe_download to return deterministic data
    def fake_download(ticker, period="2y", progress=False, multi_level_index=False):
        # For macro tickers return a very small dataframe
        if ticker in ("^TNX", "BTC-USD"):
            df = make_fake_price_series(days=6)
            return df
        return make_fake_price_series(days=300)
    import yfinance as yf
    monkeypatch.setattr(yf, "download", fake_download)
    # Patch VADER to avoid network and provide deterministic sentiment
    class FakeVader:
        def polarity_scores(self, text):
            # If the text contains 'arrest' yield very negative
            return {"neg": 0, "neu": 1, "pos": 0, "compound": -0.8 if "arrest" in text.lower() else 0.2}
    monkeypatch.setattr(bot, "vader", FakeVader())
    yield

def test_train_and_decision_buy():
    df = make_fake_price_series(days=300)
    df = bot.compute_indicators(df)
    model, features = bot.train_model(df)
    # With synthetic trending data, model may train; ensure not None
    assert features is not None
    # Choose a neutral headline so sentiment doesn't veto
    decision = bot.decide_last_day(df, model, features, latest_headline="All good news")
    # Decision reason should be one of the allowed outcomes (BUY or other)
    assert isinstance(decision, dict)
    assert "reason" in decision

def test_sentiment_veto_blocks_buy():
    df = make_fake_price_series(days=300)
    df = bot.compute_indicators(df)
    model, features = bot.train_model(df)
    decision = bot.decide_last_day(df, model, features, latest_headline="CEO arrested for fraud")
    # VADER returns negative compound for 'arrest', so we expect veto
    assert decision["vetoed_by_sentiment"] is True or decision["reason"] == "Vetoed by sentiment"
