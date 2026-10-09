"""Carga y validación de datos OHLCV (sin dependencias de modelo)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]
TIME_ALIASES = ["timestamps", "timestamp", "datetime", "date", "time", "fecha"]


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Lee un CSV OHLCV y devuelve un DataFrame ordenado con columnas
    ``timestamps, open, high, low, close[, volume, amount]``.

    Acepta nombres de columna en cualquier capitalización (p. ej. exportaciones de
    Yahoo Finance: Date, Open, High, Low, Close, Volume). ``Adj Close`` se ignora.
    ``volume``/``amount`` son opcionales.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    time_col = next((c for c in TIME_ALIASES if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No encuentro columna de fecha. Esperaba una de {TIME_ALIASES}; hay {list(df.columns)}")
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de precio: {missing}")

    df["timestamps"] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
    keep = ["timestamps"] + PRICE_COLS + [c for c in ("volume", "amount") if c in df.columns]
    df = df[keep].copy()
    num_cols = [c for c in keep if c != "timestamps"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=PRICE_COLS).sort_values("timestamps").drop_duplicates("timestamps")
    for c in ("volume", "amount"):
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    if len(df) < 2:
        raise ValueError("El CSV tiene menos de 2 filas válidas.")
    return df.reset_index(drop=True)


def select_history(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Devuelve las últimas ``lookback`` filas."""
    if lookback < 2:
        raise ValueError("lookback debe ser >= 2")
    if len(df) < lookback:
        raise ValueError(f"El CSV tiene {len(df)} filas, menos que lookback={lookback}")
    return df.iloc[-lookback:].reset_index(drop=True)


def future_timestamps(timestamps: pd.Series, pred_len: int) -> pd.Series:
    """Genera ``pred_len`` fechas futuras a partir de la frecuencia del histórico.

    Datos solo de lunes a viernes con paso diario -> días hábiles (BDay).
    """
    ts = pd.DatetimeIndex(timestamps)
    last = ts[-1]
    step = pd.Series(ts[1:] - ts[:-1]).median()
    if step == pd.Timedelta(days=1) and (ts.weekday < 5).all():
        offset = pd.offsets.BDay()
    else:
        offset = step
    future = pd.DatetimeIndex([last + offset * (i + 1) for i in range(pred_len)])
    return pd.Series(future, name="timestamps")


def synthetic_ohlcv(n: int = 600, freq: str = "5min", seed: int = 0) -> pd.DataFrame:
    """DATOS SINTÉTICOS (paseo aleatorio) SOLO PARA PRUEBAS TÉCNICAS.

    No representan ningún activo real y no sirven para sacar conclusiones financieras.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01 09:30", periods=n, freq=freq)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.001, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(10, 0.3, n)
    amount = volume * (open_ + high + low + close) / 4
    return pd.DataFrame({"timestamps": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume, "amount": amount})
