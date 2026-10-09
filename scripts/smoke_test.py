#!/usr/bin/env python
"""Prueba técnica autocontenida con DATOS SINTÉTICOS (no son un activo real).

Por defecto usa los pesos reales NeoQuasar/Kronos-small (requiere acceso a huggingface.co).
--random-init construye un Kronos minúsculo con pesos ALEATORIOS para comprobar solo la
tubería de código (sin red). Sus resultados NO tienen ningún valor predictivo y NO validan
Kronos-small.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from kronos_app.data import future_timestamps, synthetic_ohlcv  # noqa: E402
from kronos_app.inference import DEFAULT_MODEL, DEFAULT_TOKENIZER, forecast, load_predictor, rss_mb  # noqa: E402
from kronos_app.plotting import plot_forecast  # noqa: E402


def random_init_predictor():
    from model import Kronos, KronosPredictor, KronosTokenizer
    tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
                          ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=4, s2_bits=4,
                          beta=0.05, gamma0=1.0, gamma=1.1, zeta=1.0, group_size=4)
    mdl = Kronos(s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64,
                 ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, token_dropout_p=0, learn_te=True)
    return KronosPredictor(mdl.eval(), tok.eval(), device="cpu", max_context=512)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--out", default="outputs/smoke_test.png")
    a = p.parse_args()

    df = synthetic_ohlcv(a.lookback + a.horizon)
    hist, truth = df.iloc[:a.lookback], df.iloc[a.lookback:]
    fut = future_timestamps(hist["timestamps"], a.horizon)

    print("RSS inicial: %.0f MB" % rss_mb())
    predictor = random_init_predictor() if a.random_init else load_predictor(DEFAULT_MODEL, DEFAULT_TOKENIZER, device="cpu")
    print("RSS tras cargar modelo: %.0f MB" % rss_mb())
    res = forecast(predictor, hist, fut, seed=0, verbose=False)

    pred = res.prediction
    assert pred.shape == (a.horizon, 6), pred.shape
    assert np.isfinite(pred.values).all(), "hay NaN/inf en la predicción"
    assert (pred.index == fut.values).all(), "timestamps inesperados"
    print(pred.head())
    print(f"OK forma={pred.shape} | inferencia={res.seconds:.2f}s | RSS={res.rss_after_mb:.0f} MB | pico={res.peak_rss_mb:.0f} MB")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    label = "PESOS ALEATORIOS (solo test de tubería)" if a.random_init else "Kronos-small"
    plot_forecast(hist, pred, a.out, title=f"Prueba técnica con datos SINTÉTICOS - {label}",
                  note="Datos sintéticos; sin valor financiero.")
    print("Gráfico:", a.out)


if __name__ == "__main__":
    main()
