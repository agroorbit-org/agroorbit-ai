"""Engenharia de atributos para o pipeline GAIE.

Entrada: DataFrame granular (produtor × dia) de `data_loader.load_granular()`.
Saída: DataFrame com features + target `risco_seca` binário.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: F401  (mantido para tipagem)

NUM_FEATURES = [
    "temp_min",
    "temp_max",
    "chuva_mm",
    "ndvi",
    "ndwi",
    "cobertura_nuvem",
    "chuva_acum_7d",
    "chuva_acum_14d",
    "temp_amplitude",
    "ndvi_tendencia_7d",
    "dias_sem_chuva",
    "mes",
]

CAT_FEATURES = ["estado", "cultura"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "risco_seca"


def _forward_fill_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Sentinel-2 só passa a cada 5 dias — propaga última leitura para os dias entre."""
    df = df.sort_values(["produtor_id", "data_ref"]).copy()
    df["ndvi"] = df.groupby("produtor_id")["ndvi_medio"].ffill().bfill()
    df["ndwi"] = df.groupby("produtor_id")["ndwi_medio"].ffill().bfill()
    df["cobertura_nuvem"] = (
        df.groupby("produtor_id")["cobertura_nuvem"].ffill().bfill()
    )
    return df


def _add_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Janelas móveis por produtor: chuva acumulada, NDVI tendência, dias sem chuva."""
    df = df.sort_values(["produtor_id", "data_ref"]).copy()
    g = df.groupby("produtor_id", group_keys=False)

    df["chuva_acum_7d"] = g["chuva_mm"].rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
    df["chuva_acum_14d"] = g["chuva_mm"].rolling(14, min_periods=1).sum().reset_index(level=0, drop=True)

    df["temp_amplitude"] = df["temp_max"] - df["temp_min"]

    ndvi_lag7 = g["ndvi"].shift(7)
    df["ndvi_tendencia_7d"] = df["ndvi"] - ndvi_lag7
    df["ndvi_tendencia_7d"] = df["ndvi_tendencia_7d"].fillna(0)

    # dias_sem_chuva: contador sequencial reseta quando chove ≥1mm
    def _streak(s: pd.Series) -> pd.Series:
        sem_chuva = (s < 1).astype(int)
        grupo = (s >= 1).cumsum()
        return sem_chuva.groupby(grupo).cumsum()

    df["dias_sem_chuva"] = (
        df.groupby("produtor_id")["chuva_mm"]
        .apply(_streak)
        .reset_index(level=0, drop=True)
    )

    df["mes"] = df["data_ref"].dt.month

    return df


def _build_target(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Constrói `risco_seca` via função latente (sigmoide) + ruído estocástico.

    Pesos diferentes por cultura (cana e café toleram mais seca; soja menos).
    Inclui termo de ruído gaussiano para simular variabilidade real, evitando
    label leakage perfeita. Modelos têm que aprender a combinação não-linear.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    sensibilidade_cultura = {
        "soja": 1.30, "milho": 1.20, "arroz": 1.35, "algodao": 1.15,
        "cana": 0.75, "cafe": 0.70, "laranja": 0.80,
    }
    coef = df["cultura"].map(sensibilidade_cultura).fillna(1.0).values

    chuva_norm = (df["chuva_acum_14d"].values - 30) / 30  # negativo = pouca chuva
    ndvi_norm = (df["ndvi"].values - 0.45) / 0.15
    streak_norm = (df["dias_sem_chuva"].values - 5) / 5
    temp_norm = (df["temp_max"].values - 30) / 5

    # Score latente — quanto MAIOR, maior o risco
    z = coef * (-1.4 * chuva_norm - 1.0 * ndvi_norm + 0.9 * streak_norm + 0.4 * temp_norm)
    z = z + rng.normal(0, 1.2, size=len(df))  # ruído gaussiano

    prob = 1 / (1 + np.exp(-z))
    df[TARGET] = (prob > 0.5).astype(int)
    return df


def build_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de features + target.

    Espera as colunas vindas de `data_loader.load_granular()`.
    """
    df = _forward_fill_sentinel(df_raw)
    df = _add_window_features(df)
    df = _build_target(df)
    # remove linhas iniciais sem NDVI mesmo após ffill/bfill (produtor totalmente sem Sentinel)
    df = df.dropna(subset=["ndvi", "ndwi"])
    return df


if __name__ == "__main__":
    from data_loader import load_granular

    raw = load_granular()
    ds = build_dataset(raw)
    print(f"Linhas finais: {len(ds)}")
    print(f"Colunas: {list(ds.columns)}")
    print(f"Distribuição target: {ds[TARGET].value_counts(normalize=True).round(3).to_dict()}")
    print(ds[ALL_FEATURES + [TARGET]].head())
