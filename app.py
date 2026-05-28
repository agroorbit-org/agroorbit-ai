"""Entrypoint do Hugging Face Spaces — sobe a UI Gradio do AgroOrbit AI.

O Spaces executa este arquivo automaticamente. A lógica de predição,
SHAP e a interface ficam em src/predict.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predict import lancar_gradio

if __name__ == "__main__":
    lancar_gradio()
