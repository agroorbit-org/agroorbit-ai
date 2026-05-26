"""Treina, valida e compara 2 modelos (Random Forest vs XGBoost) — pipeline GAIE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_granular
from features import ALL_FEATURES, CAT_FEATURES, NUM_FEATURES, TARGET, build_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", model)])


def avaliar(nome: str, pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "modelo": nome,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n=== {nome} ===")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC AUC: {metrics['roc_auc']:.3f}  F1: {metrics['f1']:.3f}  "
          f"Prec: {metrics['precision']:.3f}  Rec: {metrics['recall']:.3f}")
    return metrics


def main():
    print("=== Pipeline GAIE — Treino e Comparação ===")
    print("\n[1/5] Carregando dados granulares do Oracle...")
    raw = load_granular()
    print(f"  {len(raw)} linhas brutas (produtor × dia)")

    print("\n[2/5] Engenharia de atributos...")
    df = build_dataset(raw)
    print(f"  {len(df)} linhas após FE — {len(ALL_FEATURES)} features + target")
    print(f"  Distribuição target: {df[TARGET].value_counts(normalize=True).round(3).to_dict()}")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    print("\n[3/5] Split treino/teste (80/20 estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")

    modelos = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
    }

    print("\n[4/5] Cross-validation (5-fold estratificado) — ROC AUC:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for nome, model in modelos.items():
        pipe = build_pipeline(model)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[nome] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "folds": [float(s) for s in scores],
        }
        print(f"  {nome:14s}: {scores.mean():.3f} ± {scores.std():.3f}")

    print("\n[5/5] Treino final + avaliação no teste:")
    resultados = []
    pipes_treinados = {}
    for nome, model in modelos.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        pipes_treinados[nome] = pipe
        resultados.append(avaliar(nome, pipe, X_test, y_test))

    # Escolha pelo ROC AUC do teste
    melhor = max(resultados, key=lambda r: r["roc_auc"])
    melhor_nome = melhor["modelo"]
    melhor_pipe = pipes_treinados[melhor_nome]

    artefato = {
        "pipeline": melhor_pipe,
        "nome": melhor_nome,
        "metricas": melhor,
        "features": ALL_FEATURES,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
    }
    joblib.dump(artefato, MODELS_DIR / "best_model.pkl")

    # Salva todas as métricas para o README
    relatorio = {
        "n_linhas_dataset": int(len(df)),
        "n_features": len(ALL_FEATURES),
        "cv_roc_auc": cv_scores,
        "teste": resultados,
        "melhor_modelo": melhor_nome,
    }
    with open(MODELS_DIR / "metricas.json", "w") as f:
        json.dump(relatorio, f, indent=2)

    print(f"\n✅ Melhor modelo: {melhor_nome} (ROC AUC teste = {melhor['roc_auc']:.3f})")
    print(f"   Salvo em: {MODELS_DIR / 'best_model.pkl'}")
    print(f"   Métricas: {MODELS_DIR / 'metricas.json'}")


if __name__ == "__main__":
    main()
