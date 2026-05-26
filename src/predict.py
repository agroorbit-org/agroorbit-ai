"""Deploy GAIE — Gradio + FastAPI servindo o melhor modelo.

Modos:
  - Gradio UI:  python src/predict.py
  - API only:   uvicorn src.predict:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent))

from features import ALL_FEATURES, CAT_FEATURES, NUM_FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
SHAP_REPORT = MODELS_DIR / "shap_report.json"

_bundle_cache = None


def _carregar_bundle():
    global _bundle_cache
    if _bundle_cache is None:
        _bundle_cache = joblib.load(MODEL_PATH)
    return _bundle_cache


def _explicar_local(pipe, X: pd.DataFrame, top_n: int = 5) -> dict:
    """SHAP local rápido para 1 amostra (TreeExplainer)."""
    import shap

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    X_t = pre.transform(X)
    nomes = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    contrib = pd.Series(sv[0], index=nomes).sort_values(key=np.abs, ascending=False)
    return {k: round(float(v), 4) for k, v in contrib.head(top_n).to_dict().items()}


def predizer_via_features(features: dict) -> dict:
    """Recebe dict com TODAS as features de ALL_FEATURES e retorna predição + SHAP."""
    bundle = _carregar_bundle()
    pipe = bundle["pipeline"]
    X = pd.DataFrame([{k: features[k] for k in ALL_FEATURES}])
    prob = float(pipe.predict_proba(X)[0, 1])
    label = int(prob > 0.5)
    return {
        "modelo": bundle["nome"],
        "risco_seca_prob": round(prob, 3),
        "risco_seca_label": label,
        "classificacao": "ALTO RISCO" if label else "OK",
        "shap_top": _explicar_local(pipe, X),
    }


def predizer_via_produtor(produtor_id: str) -> dict:
    """Busca leituras dos últimos 30 dias do produtor, gera features e prediz."""
    from data_loader import load_granular
    from features import build_dataset

    raw = load_granular()
    raw = raw[raw["produtor_id"] == produtor_id]
    if raw.empty:
        raise ValueError(f"Produtor {produtor_id} não encontrado ou sem leituras.")

    df = build_dataset(raw)
    if df.empty:
        raise ValueError(f"Produtor {produtor_id} sem features computáveis (faltam Sentinel).")

    # Pega a linha mais recente
    ultima = df.sort_values("data_ref").iloc[[-1]][ALL_FEATURES]
    bundle = _carregar_bundle()
    pipe = bundle["pipeline"]
    prob = float(pipe.predict_proba(ultima)[0, 1])
    return {
        "produtor_id": produtor_id,
        "data_ref": str(df["data_ref"].max().date()),
        "modelo": bundle["nome"],
        "risco_seca_prob": round(prob, 3),
        "risco_seca_label": int(prob > 0.5),
        "classificacao": "ALTO RISCO" if prob > 0.5 else "OK",
        "shap_top": _explicar_local(pipe, ultima),
        "features_usadas": {k: float(ultima.iloc[0][k]) if k in NUM_FEATURES else str(ultima.iloc[0][k])
                            for k in ALL_FEATURES},
    }


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="AgroOrbit AI — Risco de Seca", version="1.0")


class PredictManualIn(BaseModel):
    estado: str
    cultura: str
    temp_min: float
    temp_max: float
    chuva_mm: float
    ndvi: float
    ndwi: float
    cobertura_nuvem: float
    chuva_acum_7d: float
    chuva_acum_14d: float
    temp_amplitude: float
    ndvi_tendencia_7d: float
    dias_sem_chuva: int
    mes: int


class PredictProdutorIn(BaseModel):
    produtor_id: str


@app.get("/health")
async def health():
    bundle = _carregar_bundle()
    return {"status": "ok", "modelo": bundle["nome"], "auc": bundle["metricas"]["roc_auc"]}


@app.post("/predict")
async def predict_endpoint(payload: PredictProdutorIn):
    try:
        return predizer_via_produtor(payload.produtor_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/predict/manual")
async def predict_manual_endpoint(payload: PredictManualIn):
    return predizer_via_features(payload.model_dump())


@app.get("/shap/global")
async def shap_global():
    return json.loads(SHAP_REPORT.read_text())


# ---------------------------------------------------------------------------
# Gradio
# ---------------------------------------------------------------------------
def _gradio_predict(estado, cultura, ndvi, ndwi, nuvem, chuva_mm, chuva_7d, chuva_14d,
                    temp_max, temp_min, dias_sem_chuva, ndvi_tend, mes):
    features = {
        "estado": estado,
        "cultura": cultura,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "temp_amplitude": temp_max - temp_min,
        "chuva_mm": chuva_mm,
        "chuva_acum_7d": chuva_7d,
        "chuva_acum_14d": chuva_14d,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndvi_tendencia_7d": ndvi_tend,
        "cobertura_nuvem": nuvem,
        "dias_sem_chuva": int(dias_sem_chuva),
        "mes": int(mes),
    }
    r = predizer_via_features(features)
    shap_str = "\n".join(f"  {k}: {v:+.3f}" for k, v in r["shap_top"].items())
    return (
        f"{r['classificacao']}  •  prob = {r['risco_seca_prob']:.1%}  •  modelo = {r['modelo']}\n\n"
        f"Top contribuições SHAP:\n{shap_str}"
    )


def lancar_gradio():
    import gradio as gr

    with gr.Blocks(title="AgroOrbit AI — GAIE") as demo:
        gr.Markdown("# 🌱 AgroOrbit AI — Risco de Seca\nPredição + explicação SHAP para pequeno produtor.")

        with gr.Row():
            estado = gr.Dropdown(
                choices=["MT", "MS", "GO", "MG", "SP", "PR", "RS", "BA", "TO", "MA", "PI", "CE", "SC"],
                value="MG", label="Estado (UF)",
            )
            cultura = gr.Dropdown(
                choices=["soja", "milho", "cana", "cafe", "arroz", "laranja", "algodao"],
                value="soja", label="Cultura",
            )

        with gr.Row():
            ndvi = gr.Slider(0, 1, value=0.45, step=0.01, label="NDVI atual")
            ndwi = gr.Slider(-0.3, 0.8, value=0.30, step=0.01, label="NDWI atual")
            nuvem = gr.Slider(0, 100, value=20, label="Cobertura nuvem (%)")

        with gr.Row():
            chuva_mm = gr.Slider(0, 80, value=1, label="Chuva hoje (mm)")
            chuva_7d = gr.Slider(0, 200, value=10, label="Chuva acumulada 7d (mm)")
            chuva_14d = gr.Slider(0, 400, value=25, label="Chuva acumulada 14d (mm)")

        with gr.Row():
            temp_max = gr.Slider(15, 45, value=30, label="Temp máx (°C)")
            temp_min = gr.Slider(5, 30, value=18, label="Temp mín (°C)")
            dias_sem_chuva = gr.Slider(0, 30, value=5, step=1, label="Dias sem chuva (seq)")

        with gr.Row():
            ndvi_tend = gr.Slider(-0.5, 0.5, value=-0.05, step=0.01, label="NDVI tendência 7d")
            mes = gr.Slider(1, 12, value=5, step=1, label="Mês")

        out = gr.Textbox(label="Resultado", lines=10)
        btn = gr.Button("Prever risco", variant="primary")
        btn.click(
            _gradio_predict,
            inputs=[estado, cultura, ndvi, ndwi, nuvem, chuva_mm, chuva_7d, chuva_14d,
                    temp_max, temp_min, dias_sem_chuva, ndvi_tend, mes],
            outputs=out,
        )

        gr.Markdown(
            "**Endpoints REST:** "
            "`POST /predict` (por `produtor_id`), "
            "`POST /predict/manual` (features), "
            "`GET /shap/global`, `GET /health`."
        )

    demo.launch(server_port=int(os.getenv("PORT", 7860)), server_name="0.0.0.0")


if __name__ == "__main__":
    lancar_gradio()
