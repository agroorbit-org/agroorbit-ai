"""Interpretabilidade SHAP — importância global + explicação local de 3 amostras.

Gera:
  - models/shap_importance.png  (bar plot)
  - models/shap_summary.png     (beeswarm)
  - models/shap_waterfall_<i>.png (3 exemplos individuais)
  - models/shap_report.json     (top features + 3 casos)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_granular
from features import ALL_FEATURES, TARGET, build_dataset

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _carregar_bundle():
    bundle = joblib.load(MODELS_DIR / "best_model.pkl")
    return bundle


def _transformar(pipe, X: pd.DataFrame):
    pre = pipe.named_steps["pre"]
    X_t = pre.transform(X)
    nomes = pre.get_feature_names_out()
    return X_t, nomes


def gerar_relatorio_shap(top_n: int = 10) -> dict:
    bundle = _carregar_bundle()
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]

    raw = load_granular()
    df = build_dataset(raw)
    X = df[ALL_FEATURES]

    # Amostra para SHAP (Tree explainer é exato; usamos 500 linhas)
    X_sample = X.sample(n=min(500, len(X)), random_state=42)
    y_sample = df.loc[X_sample.index, TARGET]
    X_trans, nomes = _transformar(pipe, X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    # Para RF binário, shap_values pode vir como list[2] ou array 3D
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # ---- Global: importância média absoluta ----
    importancia = (
        pd.Series(np.abs(sv).mean(axis=0), index=nomes)
        .sort_values(ascending=False)
    )
    top_global = importancia.head(top_n)
    print("\n=== Top features (SHAP global) ===")
    for nome, val in top_global.items():
        print(f"  {nome:35s}  {val:.4f}")

    # Bar plot
    plt.figure(figsize=(8, 5))
    top_global[::-1].plot(kind="barh", color="#2a7a3b")
    plt.title(f"SHAP — Top {top_n} features (|valor médio|)")
    plt.xlabel("|SHAP value|")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_importance.png", dpi=120)
    plt.close()

    # Beeswarm summary
    plt.figure()
    shap.summary_plot(sv, X_trans, feature_names=nomes, show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=120)
    plt.close()

    # ---- Local: 3 exemplos (1 risco alto / 1 risco baixo / 1 incerto) ----
    probas = pipe.predict_proba(X_sample)[:, 1]
    idx_alto = int(np.argmax(probas))
    idx_baixo = int(np.argmin(probas))
    idx_incerto = int(np.argmin(np.abs(probas - 0.5)))
    casos = {"alto_risco": idx_alto, "baixo_risco": idx_baixo, "incerto": idx_incerto}

    relatorio_casos = {}
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)) and np.ndim(base_value) > 0:
        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else float(np.atleast_1d(base_value)[0])

    for nome_caso, idx in casos.items():
        prob = float(probas[idx])
        amostra = X_sample.iloc[idx]
        sv_amostra = sv[idx]
        contrib = (
            pd.Series(sv_amostra, index=nomes).sort_values(key=np.abs, ascending=False)
        )
        top_local = contrib.head(5).to_dict()
        relatorio_casos[nome_caso] = {
            "produtor_id": df.loc[X_sample.index[idx], "produtor_id"],
            "estado": amostra["estado"],
            "cultura": amostra["cultura"],
            "prob_risco": round(prob, 3),
            "label_real": int(y_sample.iloc[idx]),
            "top_contribuicoes": {k: round(float(v), 4) for k, v in top_local.items()},
        }

        # Waterfall por amostra
        exp_obj = shap.Explanation(
            values=sv_amostra,
            base_values=base_value,
            data=X_trans[idx],
            feature_names=list(nomes),
        )
        plt.figure()
        shap.plots.waterfall(exp_obj, show=False, max_display=10)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / f"shap_waterfall_{nome_caso}.png", dpi=120, bbox_inches="tight")
        plt.close()

    relatorio = {
        "modelo": bundle["nome"],
        "n_amostras_shap": len(X_sample),
        "top_features_global": top_global.round(4).to_dict(),
        "casos_individuais": relatorio_casos,
    }
    with open(MODELS_DIR / "shap_report.json", "w") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    print("\n=== 3 casos individuais ===")
    for nome_caso, info in relatorio_casos.items():
        print(f"  [{nome_caso}] {info['produtor_id']} ({info['estado']}/{info['cultura']}) "
              f"prob={info['prob_risco']} real={info['label_real']}")

    print(f"\n✅ Artefatos salvos em {MODELS_DIR}")
    return relatorio


def top_features(X: pd.DataFrame, n: int = 3) -> list[str]:
    """Top N features globais (compatibilidade com chamada do bot)."""
    rel = json.loads((MODELS_DIR / "shap_report.json").read_text())
    return list(rel["top_features_global"].keys())[:n]


if __name__ == "__main__":
    gerar_relatorio_shap()
