# agroorbit-ai

> **Disciplina:** Generative AI For Engineering (GAIE) — Global Solution FIAP
> **Aluno:** Samuel Ramos de Almeida (RM99134)
> **Papel na plataforma AgroOrbit:** API de predição de risco agrícola consumida
> pelo `agroorbit-bot` (WhatsApp). Lê features do Oracle FIAP populado pelo
> `agroorbit-data` (BDDI) e retorna predição + explicação SHAP.

---

## 1. Contexto do problema

Pequeno produtor rural brasileiro perde safra por falta de informação técnica
sobre risco climático. Diariamente milhares de fazendas operam sem acesso a
modelos de risco que cooperativas e grandes players já consomem.

O **AgroOrbit AI** treina um modelo supervisionado que, para cada fazenda
cadastrada, prediz **risco de seca a curto prazo** combinando:

- Imagens orbitais Sentinel-2 (NDVI, NDWI, cobertura de nuvem)
- Clima derivado de satélite NASA POWER (chuva, temperatura)
- Localização (estado) e tipo de cultura

A predição é exposta via API REST consumida pelo bot WhatsApp e por uma UI
Gradio para demonstração.

**Conexão com Economia Espacial:** o pipeline é alimentado por duas
constelações de satélites (ESA Copernicus Sentinel-2 + NASA Earth Observing
System). Sem dados orbitais, este modelo simplesmente não existe.

---

## 2. Fonte dos dados

| Fonte | Volume | Origem |
|---|---|---|
| **Oracle FIAP** — `leituras_clima` | 1860 linhas | NASA POWER + sintético calibrado |
| **Oracle FIAP** — `leituras_sentinel` | 296 linhas | Sentinel Hub CDSE + sintético calibrado |
| **Oracle FIAP** — `produtores` | 60 fazendas | Seed real + sintético cobrindo 13 UFs |

O pipeline `agroorbit-data` (BDDI) é quem ingere os dados reais. Para atender o
requisito GAIE de **≥1000 linhas × ≥10 colunas**, o script
`scripts/expand_dataset.py` complementa com produtores sintéticos cujas
distribuições foram calibradas nos dados reais (média de chuva por UF, NDVI
típico por cultura, etc.).

**Dataset final do treino:** 2020 linhas × 14 features + 1 target — após
engenharia de atributos com janelas móveis.

---

## 3. Metodologia

### 3.1 Pré-processamento (`src/data_loader.py`)

Query SQL com `JOIN` entre `leituras_clima`, `leituras_sentinel` e `produtores`
retornando granularidade `(produtor, dia)`. Sentinel-2 só revisita a cada 5
dias, então é `LEFT JOIN` e o forward-fill acontece em pandas.

### 3.2 Engenharia de atributos (`src/features.py`)

Janelas móveis por produtor:

| Feature | Como | Por quê |
|---|---|---|
| `chuva_acum_7d`, `chuva_acum_14d` | Rolling sum 7d / 14d | Captura acumulado relevante para risco |
| `dias_sem_chuva` | Contador sequencial reseta em chuva ≥1mm | Indicador clássico de estresse hídrico |
| `temp_amplitude` | `temp_max − temp_min` | Amplitude térmica afeta cultura |
| `ndvi_tendencia_7d` | `ndvi − ndvi(t-7d)` | Tendência > nível absoluto |
| `mes` | Extraído da data | Sazonalidade |

Categóricas: `estado` (13 UFs), `cultura` (7 culturas) — `OneHotEncoder`.
Numéricas — `StandardScaler`.

### 3.3 Target

`risco_seca` (binário) construído via função latente sigmoide com pesos
diferentes por cultura (soja/milho/arroz mais sensíveis; cana/café/laranja
mais tolerantes) + ruído gaussiano — evita label leakage perfeita e força o
modelo a aprender a interação cultura × clima.

Distribuição final: 54% baixo risco / 46% alto risco (balanceado).

### 3.4 Treino e validação (`src/train.py`)

- Split 80/20 estratificado
- Cross-validation 5-fold estratificado no treino
- Métricas: ROC AUC, F1, Precision, Recall, matriz de confusão
- Pipeline scikit-learn: `ColumnTransformer` → modelo
- Seleção pelo melhor ROC AUC no teste

---

## 4. Modelos testados

| Modelo | Hiperparâmetros principais |
|---|---|
| **Random Forest** | 300 árvores, max_depth=10, min_samples_split=5 |
| **XGBoost** | 300 estimadores, max_depth=6, learning_rate=0.1 |

---

## 5. Resultados obtidos

### 5.1 Cross-validation (5-fold no treino) — ROC AUC

| Modelo | Média | Desvio | Folds |
|---|---|---|---|
| **Random Forest** | **0.948** | ±0.015 | [0.945, 0.948, 0.922, 0.957, 0.966] |
| XGBoost | 0.939 | ±0.017 | [0.943, 0.929, 0.912, 0.955, 0.956] |

### 5.2 Avaliação no teste (404 amostras)

| Métrica | Random Forest | XGBoost |
|---|---|---|
| **ROC AUC** | **0.932** | 0.921 |
| F1 | 0.827 | 0.818 |
| Precision | 0.831 | 0.813 |
| Recall | 0.822 | 0.822 |

**Matriz de confusão — Random Forest (vencedor):**

|  | Pred=Baixo | Pred=Alto |
|---|---|---|
| Real=Baixo | 194 | 30 |
| Real=Alto | 32 | 148 |

→ 84.6% acurácia, taxa de falso negativo (perda crítica para o produtor) de 17.8%.

---

## 6. Interpretação com SHAP

Implementação em `src/shap_explainer.py` — `TreeExplainer` (exato para
árvores) sobre 500 amostras.

### 6.1 Importância global (top 10)

| # | Feature | \|SHAP médio\| |
|---|---|---|
| 1 | chuva_acum_14d | 0.0880 |
| 2 | ndvi | 0.0711 |
| 3 | chuva_acum_7d | 0.0681 |
| 4 | dias_sem_chuva | 0.0567 |
| 5 | ndwi | 0.0486 |
| 6 | chuva_mm (dia) | 0.0466 |
| 7 | temp_max | 0.0316 |
| 8 | temp_amplitude | 0.0239 |
| 9 | temp_min | 0.0227 |
| 10 | cobertura_nuvem | 0.0204 |

→ Confirma intuição agronômica: **chuva acumulada 14d** domina, seguida por
NDVI e dias sem chuva. Temperatura e cobertura de nuvem têm peso menor.

Gráficos em `models/shap_importance.png` (bar) e `models/shap_summary.png`
(beeswarm).

### 6.2 Casos individuais (3 amostras)

| Caso | Produtor | UF/Cultura | Prob | Real |
|---|---|---|---|---|
| **Alto risco** | seed-go-002 | GO / milho | 99.5% | 1 |
| **Baixo risco** | seed-syn-031 | PR / café | 0.0% | 0 |
| **Incerto** | seed-syn-036 | PI / milho | 50.1% | 1 |

Waterfall plots: `models/shap_waterfall_alto_risco.png`,
`shap_waterfall_baixo_risco.png`, `shap_waterfall_incerto.png`.

Reporte completo em JSON: `models/shap_report.json`.

---

## 7. Deploy

Stack: **Gradio** (UI demo) + **FastAPI** (endpoints REST consumidos pelo bot).

Endpoints:

| Método | Rota | Função |
|---|---|---|
| GET | `/health` | status do modelo + AUC |
| POST | `/predict` | predição por `produtor_id` (consulta Oracle) |
| POST | `/predict/manual` | predição via dict de features |
| GET | `/shap/global` | top features globais + casos |

Exemplo de chamada:

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"produtor_id":"seed-mt-001"}'
```

Resposta:

```json
{
  "produtor_id": "seed-mt-001",
  "data_ref": "2026-05-25",
  "modelo": "random_forest",
  "risco_seca_prob": 0.011,
  "classificacao": "OK",
  "shap_top": {
    "num__dias_sem_chuva": 0.0694,
    "num__chuva_acum_14d": 0.0689,
    "num__chuva_mm": 0.0676
  }
}
```

---

## 8. Instruções de execução

### 8.1 Pré-requisitos

- Python 3.11
- Oracle FIAP acessível (`ORACLE_HOST=oracle.fiap.com.br`)
- Pipeline BDDI (`agroorbit-data`) já rodado pelo menos uma vez (popula
  tabelas reais)

### 8.2 Setup

```bash
cd repos/agroorbit-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# preencher ORACLE_USER e ORACLE_PASSWORD em .env
```

### 8.3 Popular dataset (≥1000 linhas no Oracle)

```bash
python scripts/expand_dataset.py
```

### 8.4 Treinar + avaliar + escolher melhor

```bash
python src/train.py
```

Artefatos gerados:

- `models/best_model.pkl` — pipeline (pré-proc + modelo)
- `models/metricas.json` — CV + teste

### 8.5 Gerar SHAP global + 3 casos

```bash
python src/shap_explainer.py
```

### 8.6 Servir Gradio + API

```bash
python src/predict.py            # UI em http://localhost:7860
# OU só API:
uvicorn src.predict:app --host 0.0.0.0 --port 7860
```

---

## 9. Estrutura do repositório

```
agroorbit-ai/
├── scripts/
│   └── expand_dataset.py        # popula Oracle com sintético calibrado
├── src/
│   ├── data_loader.py           # queries Oracle (granular + agregado)
│   ├── features.py              # engenharia de atributos + target
│   ├── train.py                 # treino + CV + comparação + seleção
│   ├── shap_explainer.py        # SHAP global + 3 casos individuais
│   └── predict.py               # Gradio + FastAPI
├── models/                      # gerados pelo pipeline
│   ├── best_model.pkl
│   ├── metricas.json
│   ├── shap_report.json
│   ├── shap_importance.png
│   ├── shap_summary.png
│   └── shap_waterfall_*.png
├── requirements.txt
├── README.md
└── .env.example
```

---

## 10. Checklist GAIE

| Critério | Peso | Status |
|---|---:|---|
| Dataset ≥1000 linhas × ≥10 colunas | 15 | ✅ 2020 × 14 |
| Pré-processamento e engenharia de atributos | 20 | ✅ rolling windows, lag, one-hot, scaler |
| Aplicação e comparação de modelos (≥2 técnicas) | 20 | ✅ RF vs XGBoost |
| Validação e análise de métricas | 15 | ✅ CV 5-fold + ROC AUC + F1 + P/R + CM |
| Interpretabilidade com SHAP | 10 | ✅ global + 3 individuais |
| Deploy da aplicação | 10 | ✅ Gradio + FastAPI |
| Organização e README | 10 | ✅ este arquivo |
| **Total** | **100** | ✅ |
