"""Carga de modelos Kronos e inferencia, con medición de tiempo y memoria."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd

DEFAULT_MODEL = "NeoQuasar/Kronos-small"
DEFAULT_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def rss_mb() -> float:
    """Memoria residente actual del proceso (MB), leída de /proc."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def peak_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB


def load_predictor(model_id: str = DEFAULT_MODEL, tokenizer_id: str = DEFAULT_TOKENIZER,
                   device: str | None = None, max_context: int = 512):
    """Descarga (o reutiliza la caché de HF) modelo y tokenizer y crea el predictor.

    ``model_id``/``tokenizer_id`` pueden ser ids de Hugging Face o carpetas locales.
    Caché: ``HF_HOME`` (por defecto ``~/.cache/huggingface``).
    """
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    model.eval()
    tokenizer.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


@dataclass
class ForecastResult:
    prediction: pd.DataFrame
    seconds: float
    rss_before_mb: float
    rss_after_mb: float
    peak_rss_mb: float


def forecast(predictor, history: pd.DataFrame, future_ts: pd.Series, *, T: float = 1.0,
             top_p: float = 0.9, sample_count: int = 1, seed: int | None = None,
             verbose: bool = False) -> ForecastResult:
    """Ejecuta una predicción. ``history`` lleva columna ``timestamps`` + OHLC[+volume/amount]."""
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    x_df = history[[c for c in FEATURES if c in history.columns]].reset_index(drop=True)
    before = rss_mb()
    t0 = time.perf_counter()
    pred = predictor.predict(
        df=x_df,
        x_timestamp=history["timestamps"].reset_index(drop=True),
        y_timestamp=future_ts.reset_index(drop=True),
        pred_len=len(future_ts), T=T, top_p=top_p, sample_count=sample_count, verbose=verbose,
    )
    secs = time.perf_counter() - t0
    return ForecastResult(pred, secs, before, rss_mb(), peak_rss_mb())
