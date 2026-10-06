import numpy as np
import pandas as pd
import pytest

from task1_financial.indicators import add_all_indicators, bollinger_bands, ema, macd, rsi, sma

# StockCharts "RSI" ChartSchool worked example (Wilder, 14-period).
STOCKCHARTS_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
    46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18,
    44.22, 44.57, 43.42, 42.66, 43.13,
]
STOCKCHARTS_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38, 54.71, 50.42, 39.99, 41.46,
    41.87, 45.46, 37.30, 33.09, 37.79,
]


@pytest.fixture
def prices() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))


def naive_ema(values, span):
    alpha = 2 / (span + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return np.array(out)


def test_rsi_matches_stockcharts_reference():
    got = rsi(pd.Series(STOCKCHARTS_CLOSES)).dropna().to_numpy()
    assert len(got) == len(STOCKCHARTS_RSI)
    # reference sheet rounds intermediate averages; difference must shrink to rounding error
    np.testing.assert_allclose(got, STOCKCHARTS_RSI, atol=0.1)
    np.testing.assert_allclose(got[-2:], STOCKCHARTS_RSI[-2:], atol=0.01)


def test_rsi_bounds_and_edge_cases(prices):
    values = rsi(prices).dropna()
    assert values.between(0, 100).all()
    assert rsi(pd.Series([10.0] * 30)).dropna().eq(50).all()
    assert rsi(pd.Series(np.arange(30, dtype=float))).dropna().eq(100).all()
    assert rsi(pd.Series([1.0, 2.0])).isna().all()


def test_sma_matches_manual_mean(prices):
    got = sma(prices, 50)
    assert got.iloc[:49].isna().all()
    assert got.iloc[-1] == pytest.approx(prices.iloc[-50:].mean())


def test_ema_matches_recursive_definition(prices):
    np.testing.assert_allclose(ema(prices, 12).to_numpy(), naive_ema(prices.to_numpy(), 12))


def test_macd_matches_recursive_definition(prices):
    result = macd(prices)
    line = naive_ema(prices.to_numpy(), 12) - naive_ema(prices.to_numpy(), 26)
    signal = naive_ema(line[25:], 9)
    assert result["macd"].iloc[-1] == pytest.approx(line[-1])
    assert result["macd_signal"].iloc[-1] == pytest.approx(signal[-1])
    assert result["macd_hist"].iloc[-1] == pytest.approx(line[-1] - signal[-1])
    assert result["macd_signal"].first_valid_index() == 26 + 9 - 2


def test_bollinger_bands_use_population_std(prices):
    bb = bollinger_bands(prices)
    window = prices.iloc[-20:].to_numpy()
    mid, std = window.mean(), window.std(ddof=0)
    assert bb["bb_mid"].iloc[-1] == pytest.approx(mid)
    assert bb["bb_upper"].iloc[-1] == pytest.approx(mid + 2 * std)
    assert bb["bb_lower"].iloc[-1] == pytest.approx(mid - 2 * std)


def test_add_all_indicators_columns(prices):
    df = add_all_indicators(pd.DataFrame({"Close": prices}))
    expected = {"sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_pct_b"}
    assert expected <= set(df.columns)
    assert df.iloc[-1][list(expected)].notna().all()
