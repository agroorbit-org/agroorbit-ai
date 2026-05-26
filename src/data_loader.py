"""Lê dados granulares (produtor × data) do Oracle FIAP para o pipeline GAIE."""

from __future__ import annotations

import os
from pathlib import Path

import oracledb
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_connection():
    return oracledb.connect(
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=f"{os.environ['ORACLE_HOST']}:{os.environ['ORACLE_PORT']}/{os.environ['ORACLE_SID']}",
    )


# Granular: 1 linha por (produtor, dia). Sentinel é forward-filled em features.py
QUERY_GRANULAR = """
SELECT
  lc.produtor_id,
  lc.data_ref,
  p.estado,
  p.cultura,
  p.lat,
  p.lon,
  lc.temp_min,
  lc.temp_max,
  lc.chuva_mm,
  ls.ndvi_medio,
  ls.ndwi_medio,
  ls.cobertura_nuvem
FROM leituras_clima lc
JOIN produtores p
  ON p.id = lc.produtor_id
LEFT JOIN leituras_sentinel ls
  ON ls.produtor_id = lc.produtor_id
 AND ls.data_imagem = lc.data_ref
ORDER BY lc.produtor_id, lc.data_ref
"""


def load_granular() -> pd.DataFrame:
    """Retorna DataFrame granular (produtor × dia) — base do dataset GAIE."""
    with get_connection() as conn:
        df = pd.read_sql(QUERY_GRANULAR, conn)
    df.columns = [c.lower() for c in df.columns]
    df["data_ref"] = pd.to_datetime(df["data_ref"])
    return df


# Agregado por produtor (uso opcional pelo /predict endpoint)
QUERY_AGREGADO_PRODUTOR = """
SELECT
  p.id              AS produtor_id,
  p.estado,
  p.cultura,
  AVG(ls.ndvi_medio)        AS ndvi_medio_7d,
  AVG(ls.ndwi_medio)        AS ndwi_medio_7d,
  AVG(ls.cobertura_nuvem)   AS nuvem_media,
  SUM(lc.chuva_mm)          AS chuva_7d,
  AVG(lc.temp_max)          AS temp_max_media,
  AVG(lc.temp_min)          AS temp_min_media,
  COUNT(CASE WHEN lc.chuva_mm < 1 THEN 1 END) AS dias_sem_chuva
FROM produtores p
LEFT JOIN leituras_sentinel ls
  ON ls.produtor_id = p.id AND ls.data_imagem >= TRUNC(SYSDATE) - 7
LEFT JOIN leituras_clima lc
  ON lc.produtor_id = p.id AND lc.data_ref     >= TRUNC(SYSDATE) - 7
WHERE p.id = :produtor_id
GROUP BY p.id, p.estado, p.cultura
"""


def load_features_produtor(produtor_id: str) -> pd.DataFrame:
    """Features agregadas dos últimos 7 dias para 1 produtor (endpoint /predict)."""
    with get_connection() as conn:
        df = pd.read_sql(QUERY_AGREGADO_PRODUTOR, conn, params={"produtor_id": produtor_id})
    df.columns = [c.lower() for c in df.columns]
    return df


if __name__ == "__main__":
    df = load_granular()
    print(f"Linhas: {len(df)}")
    print(f"Colunas: {list(df.columns)}")
    print(df.head())
