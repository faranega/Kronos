#!/usr/bin/env python
"""Descarga modelo y tokenizer de Hugging Face a la caché local (HF_HOME)."""
import sys

from huggingface_hub import snapshot_download

IDS = sys.argv[1:] or ["NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small"]
for repo in IDS:
    print(f"Descargando {repo} ...")
    print("  ->", snapshot_download(repo_id=repo))
