"""Carrega o dataset sintético em Excel para o pipeline GAIE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "agroorbit_dataset.xlsx"

REQUIRED_COLUMNS = [
    "produtor_id",
    "data_ref",
    "estado",
    "cultura",
    "lat",
    "lon",
    "temp_min",
    "temp_max",
    "chuva_mm",
    "ndvi_medio",
    "ndwi_medio",
    "cobertura_nuvem",
]


def dataset_path() -> Path:
    return DATASET_PATH


def _normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset sem colunas obrigatorias: {missing}")
    df = df[REQUIRED_COLUMNS]
    df["data_ref"] = pd.to_datetime(df["data_ref"])
    df["produtor_id"] = df["produtor_id"].astype(str)
    df["estado"] = df["estado"].astype(str)
    df["cultura"] = df["cultura"].astype(str)
    numeric_cols = [
        "lat",
        "lon",
        "temp_min",
        "temp_max",
        "chuva_mm",
        "ndvi_medio",
        "ndwi_medio",
        "cobertura_nuvem",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=REQUIRED_COLUMNS)


def load_granular(path: str | Path | None = None) -> pd.DataFrame:
    """Retorna DataFrame granular produtor x dia a partir do Excel local."""
    path = Path(path) if path else dataset_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset Excel nao encontrado em {path}. "
            "Rode scripts/expand_dataset.py para gerar a planilha."
        )
    df = pd.read_excel(path, sheet_name="leituras_agroorbit")
    return _normalizar_colunas(df)


def load_features_produtor(produtor_id: str, path: str | Path | None = None) -> pd.DataFrame:
    """Agrega as leituras dos ultimos 7 dias de um produtor para uso opcional."""
    df = load_granular(path)
    df = df[df["produtor_id"] == produtor_id].sort_values("data_ref")
    if df.empty:
        return df
    janela = df.tail(7)
    return pd.DataFrame(
        [
            {
                "produtor_id": produtor_id,
                "estado": janela["estado"].iloc[-1],
                "cultura": janela["cultura"].iloc[-1],
                "ndvi_medio_7d": janela["ndvi_medio"].mean(),
                "ndwi_medio_7d": janela["ndwi_medio"].mean(),
                "nuvem_media": janela["cobertura_nuvem"].mean(),
                "chuva_7d": janela["chuva_mm"].sum(),
                "temp_max_media": janela["temp_max"].mean(),
                "temp_min_media": janela["temp_min"].mean(),
                "dias_sem_chuva": int((janela["chuva_mm"] < 1).sum()),
            }
        ]
    )


if __name__ == "__main__":
    df = load_granular()
    print(f"Arquivo: {dataset_path()}")
    print(f"Linhas: {len(df)}")
    print(f"Colunas: {list(df.columns)}")
    print(df.head())
