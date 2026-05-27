"""Gera o Excel sintético do AgroOrbit AI sem usar Oracle.

Saída padrão:
  data/agroorbit_dataset.xlsx

O arquivo atende ao requisito GAIE de dataset sintético com pelo menos
1.000 linhas e 10 colunas. O padrão atual gera 1.500 linhas e 12 colunas.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "agroorbit_dataset.xlsx"
SEED = 42

ESTADOS = {
    "MT": {"chuva": 3.0, "tmin": 22, "tmax": 33, "lat": -12.5, "lon": -55.5},
    "MS": {"chuva": 2.5, "tmin": 20, "tmax": 32, "lat": -21.6, "lon": -55.0},
    "GO": {"chuva": 2.2, "tmin": 21, "tmax": 32, "lat": -17.8, "lon": -50.9},
    "MG": {"chuva": 2.0, "tmin": 18, "tmax": 30, "lat": -18.5, "lon": -46.5},
    "SP": {"chuva": 2.4, "tmin": 19, "tmax": 30, "lat": -21.1, "lon": -47.8},
    "PR": {"chuva": 3.5, "tmin": 16, "tmax": 28, "lat": -24.9, "lon": -53.4},
    "RS": {"chuva": 3.2, "tmin": 14, "tmax": 26, "lat": -30.0, "lon": -53.5},
    "BA": {"chuva": 1.2, "tmin": 22, "tmax": 34, "lat": -12.0, "lon": -41.0},
    "TO": {"chuva": 3.0, "tmin": 23, "tmax": 34, "lat": -10.0, "lon": -48.5},
    "MA": {"chuva": 2.5, "tmin": 23, "tmax": 35, "lat": -5.5, "lon": -45.0},
    "PI": {"chuva": 1.0, "tmin": 24, "tmax": 36, "lat": -7.0, "lon": -42.0},
    "CE": {"chuva": 0.8, "tmin": 24, "tmax": 35, "lat": -5.0, "lon": -39.0},
    "SC": {"chuva": 3.8, "tmin": 15, "tmax": 26, "lat": -27.0, "lon": -50.0},
}

CULTURAS = {
    "soja": {"ndvi": 0.55, "variacao": 0.16, "sens": 1.30},
    "milho": {"ndvi": 0.50, "variacao": 0.18, "sens": 1.20},
    "cana": {"ndvi": 0.65, "variacao": 0.10, "sens": 0.75},
    "cafe": {"ndvi": 0.70, "variacao": 0.09, "sens": 0.70},
    "arroz": {"ndvi": 0.45, "variacao": 0.20, "sens": 1.35},
    "laranja": {"ndvi": 0.68, "variacao": 0.08, "sens": 0.80},
    "algodao": {"ndvi": 0.52, "variacao": 0.16, "sens": 1.15},
}


def gerar_dataset(n_produtores: int = 50, dias: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ufs = list(ESTADOS)
    culturas = list(CULTURAS)
    inicio = date(2026, 4, 26)
    rows = []

    for idx in range(n_produtores):
        uf = ufs[idx % len(ufs)]
        cultura = culturas[idx % len(culturas)]
        est = ESTADOS[uf]
        cult = CULTURAS[cultura]
        produtor_id = f"xlsx-syn-{idx + 1:03d}"
        lat = est["lat"] + rng.normal(0, 0.7)
        lon = est["lon"] + rng.normal(0, 0.7)
        seca_restante = int(rng.integers(5, 16)) if rng.random() < 0.35 else 0
        ndvi_base = cult["ndvi"] + rng.normal(0, 0.03)

        for dia in range(dias):
            data_ref = inicio + timedelta(days=dia)
            if seca_restante > 0:
                chuva = rng.exponential(0.25)
                seca_restante -= 1
            else:
                chuva = rng.exponential(est["chuva"])
                if rng.random() < 0.09:
                    seca_restante = int(rng.integers(4, 13))

            temp_min = est["tmin"] + rng.normal(0, 1.8)
            temp_max = max(temp_min + 4, est["tmax"] + rng.normal(0, 2.3))
            chuva_7d_proxy = chuva + rng.exponential(est["chuva"] * 4)
            fator_agua = np.clip(chuva_7d_proxy / 30, 0, 1)
            stress = np.clip((0.55 - fator_agua) * cult["sens"], -0.25, 0.45)
            ndvi = np.clip(
                ndvi_base + rng.normal(0, cult["variacao"] * 0.22) - stress * 0.18,
                0.05,
                0.92,
            )
            ndwi = np.clip(ndvi - 0.12 + rng.normal(0, 0.06) + fator_agua * 0.08, -0.3, 0.8)
            nuvem = np.clip(100 * rng.random() ** 1.8, 0, 99)

            rows.append(
                {
                    "produtor_id": produtor_id,
                    "data_ref": data_ref,
                    "estado": uf,
                    "cultura": cultura,
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                    "temp_min": round(float(temp_min), 2),
                    "temp_max": round(float(temp_max), 2),
                    "chuva_mm": round(float(min(chuva, 90)), 2),
                    "ndvi_medio": round(float(ndvi), 3),
                    "ndwi_medio": round(float(ndwi), 3),
                    "cobertura_nuvem": round(float(nuvem), 2),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    df = gerar_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="leituras_agroorbit", index=False)
        resumo = pd.DataFrame(
            {
                "metrica": ["linhas", "colunas", "produtores", "dias_por_produtor"],
                "valor": [len(df), len(df.columns), df["produtor_id"].nunique(), 30],
            }
        )
        resumo.to_excel(writer, sheet_name="resumo", index=False)

    print(f"Excel gerado em: {OUTPUT_PATH}")
    print(f"Linhas: {len(df)} | Colunas: {len(df.columns)}")


if __name__ == "__main__":
    main()
