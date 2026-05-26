"""
Popula Oracle FIAP com produtores e leituras sintéticas calibradas
nas distribuições dos dados reais — garante ≥1000 linhas granulares
(produtor × data) para o pipeline GAIE.

Idempotente: remove `seed-syn-*` antes de reinserir.
"""

from __future__ import annotations

import os
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import oracledb
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Calibração — distribuições baseadas nos dados reais já no Oracle
#   NDVI: 0.05..0.74 (média ~0.30)
#   chuva: 0..40 mm/dia (média ~1.2)
#   temp: 10..38 °C
#   cobertura nuvem: 0..95%
# ---------------------------------------------------------------------------
CULTURAS = {
    "soja": {"ndvi_base": 0.55, "ndvi_var": 0.18},
    "milho": {"ndvi_base": 0.50, "ndvi_var": 0.20},
    "cana": {"ndvi_base": 0.65, "ndvi_var": 0.12},
    "cafe": {"ndvi_base": 0.70, "ndvi_var": 0.10},
    "arroz": {"ndvi_base": 0.45, "ndvi_var": 0.22},
    "laranja": {"ndvi_base": 0.68, "ndvi_var": 0.10},
    "algodao": {"ndvi_base": 0.52, "ndvi_var": 0.18},
}

ESTADOS = {
    # UF: (chuva_mm_média_dia, temp_min_base, temp_max_base)
    "MT": (3.0, 22, 33),
    "MS": (2.5, 20, 32),
    "GO": (2.2, 21, 32),
    "MG": (2.0, 18, 30),
    "SP": (2.4, 19, 30),
    "PR": (3.5, 16, 28),
    "RS": (3.2, 14, 26),
    "BA": (1.2, 22, 34),
    "TO": (3.0, 23, 34),
    "MA": (2.5, 23, 35),
    "PI": (1.0, 24, 36),
    "CE": (0.8, 24, 35),
    "SC": (3.8, 15, 26),
}

NOMES = [
    "Aline",
    "Bruno",
    "Carla",
    "Diego",
    "Elisa",
    "Fábio",
    "Gabi",
    "Heitor",
    "Iara",
    "Júlio",
    "Kátia",
    "Lucas",
    "Mariana",
    "Nuno",
    "Olívia",
    "Paulo",
    "Quésia",
    "Rafael",
    "Sandra",
    "Tiago",
    "Úrsula",
    "Vitor",
    "Wanda",
    "Yuri",
    "Zeca",
    "Alice",
    "Bento",
    "Clara",
    "Davi",
    "Elena",
]

N_PRODUTORES_SINT = 40
DIAS_HISTORICO = 30


def gera_produtores_sinteticos():
    out = []
    estados_list = list(ESTADOS.keys())
    culturas_list = list(CULTURAS.keys())
    for i in range(N_PRODUTORES_SINT):
        uf = estados_list[i % len(estados_list)]
        cultura = culturas_list[i % len(culturas_list)]
        nome = f"{NOMES[i % len(NOMES)]} {uf}-{i:02d}"
        lat_base = {
            "MT": -12.5,
            "MS": -21.6,
            "GO": -17.8,
            "MG": -18.5,
            "SP": -21.1,
            "PR": -24.9,
            "RS": -30.0,
            "BA": -12.0,
            "TO": -10.0,
            "MA": -5.5,
            "PI": -7.0,
            "CE": -5.0,
            "SC": -27.0,
        }[uf]
        lon_base = {
            "MT": -55.5,
            "MS": -55.0,
            "GO": -50.9,
            "MG": -46.5,
            "SP": -47.8,
            "PR": -53.4,
            "RS": -53.5,
            "BA": -41.0,
            "TO": -48.5,
            "MA": -45.0,
            "PI": -42.0,
            "CE": -39.0,
            "SC": -50.0,
        }[uf]
        lat = lat_base + random.uniform(-1.5, 1.5)
        lon = lon_base + random.uniform(-1.5, 1.5)
        out.append(
            (f"seed-syn-{i:03d}", nome, round(lat, 6), round(lon, 6), uf, cultura)
        )
    return out


def gera_leituras_clima(produtor_id: str, uf: str, dia_inicial: date):
    chuva_med, tmin_base, tmax_base = ESTADOS[uf]
    leituras = []
    # estado seca: 30% prob, dias_secos sequenciais 4..14
    em_seca = random.random() < 0.30
    dias_seca_restantes = random.randint(4, 14) if em_seca else 0
    for d in range(DIAS_HISTORICO):
        data_ref = dia_inicial + timedelta(days=d)
        if dias_seca_restantes > 0:
            chuva = max(0.0, np.random.exponential(0.2))
            dias_seca_restantes -= 1
        else:
            chuva = max(0.0, np.random.exponential(chuva_med))
            # chance de novo período de seca
            if random.random() < 0.10:
                dias_seca_restantes = random.randint(3, 10)
        tmin = tmin_base + np.random.normal(0, 2)
        tmax = tmax_base + np.random.normal(0, 2.5)
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        leituras.append(
            (
                produtor_id,
                data_ref,
                round(float(tmin), 2),
                round(float(tmax), 2),
                round(float(min(chuva, 80)), 2),
            )
        )
    return leituras


def gera_leituras_sentinel(
    produtor_id: str, cultura: str, dia_inicial: date, historico_chuva: dict
):
    ndvi_base = CULTURAS[cultura]["ndvi_base"]
    ndvi_var = CULTURAS[cultura]["ndvi_var"]
    leituras = []
    # Revisita Sentinel-2 = 5 dias
    for d in range(0, DIAS_HISTORICO, 5):
        data_img = dia_inicial + timedelta(days=d)
        chuva_acum_recent = sum(
            historico_chuva.get(dia_inicial + timedelta(days=d - k), 0.0)
            for k in range(1, 8)
        )
        # NDVI cai com seca, sobe com chuva acumulada
        fator_chuva = np.clip(chuva_acum_recent / 30.0, 0, 1)
        ndvi = ndvi_base + np.random.normal(0, ndvi_var * 0.3)
        ndvi += (fator_chuva - 0.4) * 0.15
        ndvi = float(np.clip(ndvi, 0.0, 0.95))
        ndwi = float(np.clip(ndvi - 0.10 + np.random.normal(0, 0.05), -0.3, 0.8))
        nuvem = float(np.clip(np.random.beta(2, 5) * 100, 0, 99))
        leituras.append(
            (produtor_id, data_img, round(ndvi, 3), round(ndwi, 3), round(nuvem, 2))
        )
    return leituras


def conectar():
    return oracledb.connect(
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=f"{os.environ['ORACLE_HOST']}:{os.environ['ORACLE_PORT']}/{os.environ['ORACLE_SID']}",
    )


def limpar_sinteticos(cur):
    cur.execute("DELETE FROM leituras_clima    WHERE produtor_id LIKE 'seed-syn-%'")
    cur.execute("DELETE FROM leituras_sentinel WHERE produtor_id LIKE 'seed-syn-%'")
    cur.execute("DELETE FROM produtores        WHERE id LIKE 'seed-syn-%'")


def main():
    dia_final = date.today()
    dia_inicial = dia_final - timedelta(days=DIAS_HISTORICO - 1)
    produtores = gera_produtores_sinteticos()

    print(f"Conectando Oracle como {os.environ['ORACLE_USER']}...")
    with conectar() as conn:
        cur = conn.cursor()

        print("Limpando sintéticos anteriores (idempotente)...")
        limpar_sinteticos(cur)

        print(f"Inserindo {len(produtores)} produtores sintéticos...")
        cur.executemany(
            "INSERT INTO produtores (id, nome, lat, lon, estado, cultura, criado_em) "
            "VALUES (:1, :2, :3, :4, :5, :6, SYSTIMESTAMP)",
            produtores,
        )

        total_clima = 0
        total_sentinel = 0
        for pid, _, _, _, uf, cultura in produtores:
            clima = gera_leituras_clima(pid, uf, dia_inicial)
            cur.executemany(
                "INSERT INTO leituras_clima "
                "(produtor_id, data_ref, temp_min, temp_max, chuva_mm) "
                "VALUES (:1, :2, :3, :4, :5)",
                clima,
            )
            total_clima += len(clima)
            historico = {row[1]: row[4] for row in clima}
            sentinel = gera_leituras_sentinel(pid, cultura, dia_inicial, historico)
            cur.executemany(
                "INSERT INTO leituras_sentinel "
                "(produtor_id, data_imagem, ndvi_medio, ndwi_medio, cobertura_nuvem) "
                "VALUES (:1, :2, :3, :4, :5)",
                sentinel,
            )
            total_sentinel += len(sentinel)

        conn.commit()
        print(f"  +{total_clima} linhas em leituras_clima")
        print(f"  +{total_sentinel} linhas em leituras_sentinel")

        # Sumário final
        for tbl, col in [
            ("produtores", "id"),
            ("leituras_clima", "produtor_id"),
            ("leituras_sentinel", "produtor_id"),
        ]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            total = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {col} LIKE 'seed-syn-%'")
            sint = cur.fetchone()[0]
            print(f"  {tbl}: {total} total ({sint} sintéticos, {total - sint} reais)")


if __name__ == "__main__":
    main()
