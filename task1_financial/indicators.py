import numpy as np
import pandas as pd

SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_NUM_STD = 2.0
RSI_MAX = 100.0
RSI_NEUTRAL = 50.0


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Recursive EMA with alpha = 2/(span+1), seeded at the first observation (standard charting convention)."""
    return series.ewm(span=span, adjust=False).mean()


def wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: seed with the simple mean of the first `period` values, then avg = (prev*(n-1) + x) / n."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if len(valid) < period:
        return out
    arr = valid.to_numpy(dtype=float)
    smoothed = np.empty(len(arr) - period + 1)
    smoothed[0] = arr[:period].mean()
    for i, x in enumerate(arr[period:], start=1):
        smoothed[i] = (smoothed[i - 1] * (period - 1) + x) / period
    out.loc[valid.index[period - 1:]] = smoothed
    return out


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0), period)
    avg_loss = wilder_smooth(-delta.clip(upper=0), period)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = RSI_MAX - RSI_MAX / (1 + rs)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), RSI_MAX)
    result = result.mask((avg_loss == 0) & (avg_gain == 0), RSI_NEUTRAL)
    return result


def macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    line.iloc[: slow - 1] = np.nan
    signal_line = ema(line.dropna(), signal).reindex(close.index)
    signal_line.iloc[: slow + signal - 2] = np.nan
    return pd.DataFrame({"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line})


def bollinger_bands(close: pd.Series, window: int = BB_WINDOW, num_std: float = BB_NUM_STD) -> pd.DataFrame:
    """Uses population std (ddof=0), matching Bollinger's definition and most charting platforms."""
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = upper - lower
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_pct_b": (close - lower) / width.replace(0, np.nan),
        "bb_bandwidth": width / mid,
    })


def add_all_indicators(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    close = out[price_col].astype(float)
    out[f"sma_{SMA_SHORT_WINDOW}"] = sma(close, SMA_SHORT_WINDOW)
    out[f"sma_{SMA_LONG_WINDOW}"] = sma(close, SMA_LONG_WINDOW)
    out[f"rsi_{RSI_PERIOD}"] = rsi(close)
    out = out.join(macd(close)).join(bollinger_bands(close))
    return out
