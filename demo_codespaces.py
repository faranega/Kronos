"""
Predicción de ejemplo con Kronos, pensada para GitHub Codespaces.

Ejecuta:  python demo_codespaces.py

1. Descarga de Hugging Face el modelo "Kronos-small" y su tokenizador.
2. Lee precios reales de 5 minutos (tests/data/regression_input.csv).
3. Usa las primeras 400 velas para predecir las 120 siguientes.
4. Guarda una tabla (prediccion.csv) y un gráfico (prediccion.png)
   comparando la predicción con lo que pasó de verdad.
"""
import matplotlib

matplotlib.use("Agg")  # en Codespaces no hay pantalla: guardamos el gráfico en un archivo
import matplotlib.pyplot as plt
import pandas as pd

from model import Kronos, KronosPredictor, KronosTokenizer

LOOKBACK = 400   # velas que el modelo "mira" hacia atrás
PRED_LEN = 120   # velas que tiene que predecir
COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def main():
    print("1/4 Descargando el modelo de Hugging Face (solo la primera vez)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    print("2/4 Leyendo los datos de ejemplo...")
    df = pd.read_csv("tests/data/regression_input.csv", parse_dates=["timestamps"])
    x_df = df.loc[:LOOKBACK - 1, COLUMNS]
    x_timestamp = df.loc[:LOOKBACK - 1, "timestamps"]
    y_timestamp = df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1, "timestamps"]

    print("3/4 Calculando la predicción (en CPU tarda un par de minutos)...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )
    pred_df.to_csv("prediccion.csv")
    print(pred_df.head())

    print("4/4 Dibujando el gráfico...")
    real = df.loc[:LOOKBACK + PRED_LEN - 1].set_index("timestamps")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(real.index, real["close"], label="Real", color="blue")
    ax1.plot(pred_df.index, pred_df["close"], label="Predicción", color="red")
    ax1.set_ylabel("Precio de cierre")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(real.index, real["volume"], label="Real", color="blue")
    ax2.plot(pred_df.index, pred_df["volume"], label="Predicción", color="red")
    ax2.set_ylabel("Volumen")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("prediccion.png", dpi=120)

    print("\n¡Listo! Abre 'prediccion.png' en el panel de archivos de la izquierda.")


if __name__ == "__main__":
    main()
