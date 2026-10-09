#!/usr/bin/env python
"""Predice con Kronos a partir de un CSV OHLCV real.

Ejemplo:
  python scripts/predict_csv.py --csv data/AAPL_1d.csv --asset AAPL --lookback 400 --horizon 30
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kronos_app.data import future_timestamps, load_ohlcv_csv, select_history  # noqa: E402
from kronos_app.inference import DEFAULT_MODEL, DEFAULT_TOKENIZER, forecast, load_predictor  # noqa: E402
from kronos_app.plotting import plot_forecast  # noqa: E402

DISCLAIMER = "Salida técnica de un modelo; no es asesoramiento ni una estrategia de inversión."


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="CSV con fecha + open/high/low/close[/volume/amount]")
    p.add_argument("--asset", default="asset", help="Nombre del activo (títulos y nombres de salida)")
    p.add_argument("--lookback", type=int, default=400, help="Nº de velas históricas (máx. útil 512)")
    p.add_argument("--horizon", type=int, default=30, help="Nº de velas a predecir")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--device", default=None, help="cpu | cuda:0 | mps (por defecto autodetecta)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--samples", type=int, default=1, help="Trayectorias que se promedian")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    # 1) datos
    df = load_ohlcv_csv(args.csv)
    lookback = args.lookback
    if lookback > 512:
        print("Aviso: lookback > 512; se recorta a 512 (max_context de Kronos-small).")
        lookback = 512
    hist = select_history(df, lookback)
    fut = future_timestamps(hist["timestamps"], args.horizon)

    # 2) inferencia
    predictor = load_predictor(args.model, args.tokenizer, device=args.device)
    res = forecast(predictor, hist, fut, T=args.temperature, top_p=args.top_p,
                   sample_count=args.samples, seed=args.seed, verbose=True)

    # 3) salida + visualización
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"{args.asset}_h{args.horizon}")
    res.prediction.to_csv(base + "_prediction.csv", index_label="timestamps")
    plot_forecast(hist, res.prediction, base + "_forecast.png",
                  title=f"{args.asset} - Kronos ({args.model})", note=DISCLAIMER)
    print(res.prediction.head())
    print(f"Tiempo de inferencia: {res.seconds:.1f}s | RSS: {res.rss_after_mb:.0f} MB | pico: {res.peak_rss_mb:.0f} MB")
    print(f"Guardado: {base}_prediction.csv, {base}_forecast.png\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
