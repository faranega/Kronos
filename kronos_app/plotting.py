"""Visualización (sin lógica de modelo)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_forecast(history: pd.DataFrame, prediction: pd.DataFrame, out_path: str,
                  title: str = "Kronos forecast", note: str | None = None, tail: int | None = None) -> str:
    """Compara histórico y predicción (precio de cierre y volumen) y guarda un PNG."""
    hist = history if tail is None else history.iloc[-tail:]
    has_vol = "volume" in hist.columns and "volume" in prediction.columns and hist["volume"].abs().sum() > 0
    fig, axes = plt.subplots(2 if has_vol else 1, 1, figsize=(10, 6 if has_vol else 4), sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(hist["timestamps"], hist["close"], color="tab:blue", label="Histórico (close)")
    ax.plot(prediction.index, prediction["close"], color="tab:red", label="Predicción (close)")
    ax.axvline(hist["timestamps"].iloc[-1], color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Cierre")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title(title)
    if has_vol:
        ax2 = axes[1, 0]
        ax2.plot(hist["timestamps"], hist["volume"], color="tab:blue", lw=0.9)
        ax2.plot(prediction.index, prediction["volume"], color="tab:red", lw=0.9)
        ax2.set_ylabel("Volumen")
        ax2.grid(alpha=0.3)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color="gray")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
