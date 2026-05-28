---
title: AgroOrbit AI
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
---

# agroorbit-ai

> **Disciplina:** Generative AI For Engineering (GAIE) — Global Solution FIAP  
> **Aluno:** Samuel Ramos de Almeida (RM99134)  
> **Papel na plataforma AgroOrbit:** modelo de predição de risco de seca com explicação SHAP.

## 1. Contexto do Problema

O AgroOrbit AI estima risco de seca para produtores rurais combinando variáveis climáticas, vegetação orbital e contexto agrícola. O objetivo é simular um pipeline de IA aplicado à economia espacial, usando sinais inspirados em Sentinel-2/NASA POWER:

- NDVI, NDWI e cobertura de nuvem;
- chuva diária e acumulados recentes;
- temperatura mínima/máxima;
- estado, cultura e coordenadas da fazenda.

## 2. Dataset

Este projeto **não usa Oracle**. O dataset está em Excel:

```text
data/agroorbit_dataset.xlsx
```

O arquivo contém uma base sintética gerada por IA/código para atender ao requisito GAIE:

| Item                  |          Valor |
| --------------------- | -------------: |
| Linhas                |          1.500 |
| Colunas originais     |             12 |
| Produtores sintéticos |             50 |
| Dias por produtor     |             30 |
| Granularidade         | produtor × dia |

Colunas do Excel:

```text
produtor_id, data_ref, estado, cultura, lat, lon,
temp_min, temp_max, chuva_mm, ndvi_medio, ndwi_medio, cobertura_nuvem
```

Para regenerar a planilha:

```bash
python scripts/expand_dataset.py
```

## 3. Metodologia

`src/data_loader.py` lê `data/agroorbit_dataset.xlsx` com pandas (caminho fixo no código, sem `.env`).

`src/features.py` cria as features finais:

| Feature                           | Como é calculada                           |
| --------------------------------- | ------------------------------------------ |
| `chuva_acum_7d`, `chuva_acum_14d` | soma móvel por produtor                    |
| `dias_sem_chuva`                  | sequência de dias com chuva menor que 1 mm |
| `temp_amplitude`                  | `temp_max - temp_min`                      |
| `ndvi_tendencia_7d`               | diferença do NDVI contra 7 dias antes      |
| `mes`                             | extraído de `data_ref`                     |

O target `risco_seca` é binário e sintético, construído por função latente com chuva, NDVI, dias sem chuva, temperatura e sensibilidade por cultura.

## 4. Modelos

Foram comparadas duas técnicas:

| Modelo        | Configuração                                        |
| ------------- | --------------------------------------------------- |
| Random Forest | 300 árvores, `max_depth=10`, `min_samples_split=5`  |
| XGBoost       | 300 estimadores, `max_depth=6`, `learning_rate=0.1` |

Pipeline:

```text
ColumnTransformer(OneHotEncoder + StandardScaler) → modelo
```

## 5. Resultados

Dataset final após engenharia de atributos: **1.500 linhas, 14 features + target**.

Cross-validation 5-fold no treino:

| Modelo        | ROC AUC médio | Desvio |
| ------------- | ------------: | -----: |
| Random Forest |         0.838 |  0.025 |
| XGBoost       |         0.820 |  0.023 |

Teste holdout de 300 amostras:

| Métrica   | Random Forest | XGBoost |
| --------- | ------------: | ------: |
| ROC AUC   |         0.809 |   0.772 |
| F1        |         0.746 |   0.722 |
| Precision |         0.748 |   0.727 |
| Recall    |         0.743 |   0.717 |

Melhor modelo: **Random Forest**.

Matriz de confusão do Random Forest:

|            | Pred=Baixo | Pred=Alto |
| ---------- | ---------: | --------: |
| Real=Baixo |        110 |        38 |
| Real=Alto  |         39 |       113 |

## 6. SHAP

`src/shap_explainer.py` gera interpretação global e local com SHAP.

Top features globais:

|   # | Feature          | SHAP médio absoluto |
| --: | ---------------- | ------------------: |
|   1 | `ndvi`           |              0.0690 |
|   2 | `dias_sem_chuva` |              0.0682 |
|   3 | `chuva_acum_7d`  |              0.0613 |
|   4 | `ndwi`           |              0.0563 |
|   5 | `chuva_acum_14d` |              0.0547 |

Artefatos:

- `models/shap_importance.png`
- `models/shap_summary.png`
- `models/shap_waterfall_alto_risco.png`
- `models/shap_waterfall_baixo_risco.png`
- `models/shap_waterfall_incerto.png`
- `models/shap_report.json`

## 7. API e Demo

Endpoints:

| Método | Rota              | Função                              |
| ------ | ----------------- | ----------------------------------- |
| GET    | `/health`         | status do modelo                    |
| POST   | `/predict`        | predição por `produtor_id` do Excel |
| POST   | `/predict/manual` | predição por features manuais       |
| GET    | `/shap/global`    | relatório SHAP global               |

Exemplo:

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"produtor_id":"xlsx-syn-001"}'
```

## 8. Como Executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerar dataset:

```bash
python scripts/expand_dataset.py
```

Treinar e avaliar:

```bash
python src/train.py
```

Gerar SHAP:

```bash
python src/shap_explainer.py
```

Servir demo/API:

```bash
python src/predict.py
# ou
uvicorn src.predict:app --host 0.0.0.0 --port 7860
```

## 9. Estrutura

```text
agroorbit-ai/
├── data/
│   └── agroorbit_dataset.xlsx
├── scripts/
│   └── expand_dataset.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── shap_explainer.py
│   └── predict.py
├── models/
│   ├── best_model.pkl
│   ├── metricas.json
│   ├── shap_report.json
│   └── shap_*.png
├── requirements.txt
└── README.md
```

## 10. Checklist GAIE

| Critério                                     | Status                       |
| -------------------------------------------- | ---------------------------- |
| Dataset sintético com mínimo de 1.000 linhas | OK: 1.500 linhas             |
| Dataset com mínimo de 10 colunas             | OK: 12 colunas               |
| Pré-processamento e engenharia de atributos  | OK                           |
| Comparação de pelo menos 2 modelos           | OK: Random Forest vs XGBoost |
| Validação e métricas                         | OK                           |
| Interpretabilidade                           | OK: SHAP                     |
| Deploy/API/demo                              | OK                           |
