# Kronos en Claude Code Cloud (CPU) — guía de uso

Capa de uso sobre el [repositorio oficial de Kronos](https://github.com/shiyu-coder/Kronos). No modifica `model/`.

> **Aviso:** las predicciones son salidas técnicas de un modelo estadístico. No son asesoramiento
> financiero ni una estrategia de inversión rentable; no se ha evaluado su rendimiento real.

## Estado de la verificación (sesión del 2026-10-09)

| Elemento | Estado |
|---|---|
| Entorno: 4 CPU, 15 GB RAM, sin GPU, ~30 GB de disco libres | Comprobado |
| Instalación de `requirements.txt` en Python 3.12 | **Comprobado** |
| Instalación en Python 3.13 | **Falla**: `pandas==2.2.2` no tiene wheel para 3.13 y no compila. Usar 3.10–3.12 |
| `from model import Kronos, KronosTokenizer, KronosPredictor` | **Comprobado** |
| Descarga de `NeoQuasar/Kronos-small` y `Kronos-Tokenizer-base` | **NO completada**: `huggingface.co` devuelve 403 del proxy de red del entorno |
| Predicción real con Kronos-small (tiempo y memoria reales) | **NO ejecutada** (depende de la descarga) |
| Tubería (datos → inferencia → gráfico) con un Kronos minúsculo de **pesos aleatorios** y datos sintéticos | Comprobada (solo valida el código; ~0,3 s, ~830 MB RSS; **sin valor predictivo**) |

Para desbloquear: en el entorno cloud, *Network access* → permitir `huggingface.co` (y
`cdn-lfs.huggingface.co`/`*.hf.co`, que HF usa para los pesos) y volver a lanzar
`python scripts/smoke_test.py`. Docs: <https://code.claude.com/docs/en/cloud-environments#network-access>.

¿Cabe Kronos-small en este entorno? Previsiblemente sí: 24,7 M parámetros (~100 MB en fp32), contexto 512,
y 15 GB de RAM. Es una estimación, no una medición; la medición real queda pendiente de la descarga.

## Instalación reproducible

```bash
bash scripts/setup.sh            # crea .venv (Python 3.12), instala dependencias y descarga modelos
source .venv/bin/activate
```

Los modelos **no** se suben al repo: se bajan de Hugging Face a la caché (`~/.cache/huggingface`, o la ruta
de `HF_HOME`) con `python scripts/download_models.py`, o automáticamente en la primera ejecución
(`from_pretrained`). Ni `.venv/`, ni cachés, ni `outputs/`, ni `data/` se versionan (ver `.gitignore`).
Cada sesión cloud nueva debe repetir `bash scripts/setup.sh` (o ponerlo como *setup script* del entorno).

## Uso

1. Prueba técnica autocontenida (datos **sintéticos**, no un activo real):
   ```bash
   python scripts/smoke_test.py            # con Kronos-small real (necesita HF)
   python scripts/smoke_test.py --random-init   # solo tubería, sin red, pesos aleatorios
   ```
2. Predicción con tus datos reales (CSV con fecha + open/high/low/close, opcional volume/amount;
   admite cabeceras tipo Yahoo Finance):
   ```bash
   python scripts/predict_csv.py --csv data/BTC-USD.csv --asset BTC --lookback 400 --horizon 30
   ```
   Opciones: `--lookback` (máx. 512), `--horizon`, `--samples` (trayectorias promediadas), `--temperature`,
   `--top-p`, `--seed`, `--device`, `--model`, `--tokenizer`, `--out-dir`.
   Genera en `outputs/`: `<activo>_h<horizonte>_prediction.csv` y `..._forecast.png`.

Los datos de velas diarias solo de lunes a viernes se extienden en días hábiles; cualquier otro caso usa la
mediana del intervalo histórico (no conoce festivos de bolsa).

## Estructura

- `kronos_app/data.py` — carga/validación de CSV y fechas futuras (sin modelo)
- `kronos_app/inference.py` — carga de modelo, predicción, tiempo y memoria
- `kronos_app/plotting.py` — gráfico histórico vs. predicción
- `scripts/` — `setup.sh`, `download_models.py`, `smoke_test.py`, `predict_csv.py`
