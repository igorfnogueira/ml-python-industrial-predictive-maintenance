Language / Idioma: [English](README.md) | **Português**

# Relatório de Análise e Modelagem para Manutenção Preditiva

Este projeto desenvolve modelos de manutenção preditiva para identificar cinco tipos de falha em máquinas a partir de sensores, com preparação de dados, análise exploratória e classificação multi-rótulo (um classificador binário por falha).

**Dataset:** `bootcamp_train.csv` — 35.260 linhas × 15 colunas (schema no estilo AI4I: `tipo` L/M/H, temperaturas, torque, desgaste, etc., com `umidade_relativa` adicional). Origem formal (UCI/Kaggle/planta real) não está citada nos artefatos do repositório.

## 1. Preparação e Limpeza dos Dados

- **Identificação e Tratamento de Inconsistências:** Valores de falha com diferentes grafias (e.g., 'sim', 'Sim', 'True', 'não', 'Não', 'False', 'N', 'y', '1', '0') foram padronizados para 1 (falha) ou 0 (sem falha). Valores anômalos como '-' foram mapeados para 0, assumindo ausência de falha.

- **Tratamento de Nulos e Anomalias Físicas:** Valores negativos nas colunas de temperatura (em Kelvin) foram substituídos por `NaN` para posterior imputação, pois são fisicamente impossíveis. Após a padronização, as colunas binárias de falha foram convertidas para o tipo inteiro. Nulos em FDF/FA foram preenchidos com 0; demais nulos numéricos foram imputados com a mediana no pipeline de pré-processamento.

- **Gestão de Outliers e Sinais Operacionais:** Outliers e valores negativos em `velocidade_rotacional` e `desgaste_da_ferramenta` foram **mantidos** por hipótese de sinal operacional (não ruído). Para `umidade_relativa`, média/moda/mediana coincidentes e baixo desvio padrão também motivaram manter os pontos extremos. Uma comparação A/B numérica (com vs sem outliers) **não está versionada** neste repositório — a decisão permanece como hipótese de domínio.

## 2. Análise Exploratória e Visualização

- **Estatísticas Descritivas:** Calculadas para as colunas numéricas (média, mediana, moda, desvio padrão).

- **Análise de Relações:** Boxplots relacionando atributos dos sensores às falhas. Exemplos: `desgaste_da_ferramenta` como preditor forte de `FDF`; `torque` e `desgaste_da_ferramenta` relevantes para `FTE`.

## 3. Metodologia de Modelagem e Treinamento

- **Estratégia de Classificação:** Multi-rótulo via 5 classificadores binários independentes (uma falha por modelo).

- **Pipeline de Pré-processamento:** `ColumnTransformer` com imputação (mediana), `MinMaxScaler` e `OneHotEncoder` em categóricas.

- **Modelos atribuídos no experimento oficial** (`Projeto_Final_do_Bootcamp_CDIA.ipynb`):
  - `RandomForestClassifier`: FDF, FP, FA
  - `GradientBoostingClassifier`: FDC, FTE  
  Uma tabela formal RF vs GB por falha (head-to-head) **não está versionada** no notebook/scripts; a atribuição acima é a que foi executada e reportada.

- **Avaliação:** Holdout estratificado 80/20 (`test_size=0.2`, `stratify=y`, `random_state=42`). Métrica headline = **F1 da classe positiva** no conjunto de **teste**.

- **Otimização de Hiperparâmetros:** `RandomizedSearchCV` apenas no treino (`cv=5`, `scoring='f1'`, `n_iter=10`, `random_state=42`). Uma única run com seed fixa; tabela completa de trials não foi publicada.

- **Classes desbalanceadas:** `SMOTE` aplicado **somente no treino** e **somente** para `FA (Falha Aleatoria)`. O teste permanece sem reamostragem.

- **Ressalva de features (notebook oficial):** Em `train_and_evaluate_model` do notebook, ao montar `X` removem-se `id`, `id_produto` e o alvo atual, mas **permanecem** `falha_maquina` e as outras colunas de tipo de falha — risco de *label leakage* em cenário multi-rótulo. Os scripts em `Projeto/src/` usam lógica de drop ligeiramente diferente; as métricas abaixo vêm do notebook.

## 4. Resultados e Conclusão

Métricas oficiais no **holdout de teste** (7.052 amostras), F1 da classe positiva:

| Falha | Modelo | F1 (classe 1) | Positivos no teste | F1 CV (treino) |
|---|---|---|---|---|
| FDF (Desgaste Ferramenta) | Random Forest | **0,67** | 14 | 0,56 |
| FDC (Dissipação de Calor) | Gradient Boosting | **0,88** | 45 | 0,90 |
| FP (Potência) | Random Forest | **0,71** | 25 | 0,78 |
| FTE (Tensão Excessiva) | Gradient Boosting | **0,89** | 34 | 0,74 |
| FA (Aleatória) | Random Forest + SMOTE | **0,00** | 15 | 0,996* |

\*CV de FA no treino com SMOTE — otimista frente ao F1=0,00 no teste. Os 15/7.052 são contagens do **teste**, não a prevalência no dataset completo (~ordem de 75/35.260 para FA).

- **Resumo:** F1 entre **0,67 e 0,89** em 4 das 5 falhas (melhor: FTE = 0,89). FA não convergiu (classe extremamente rara). Accuracy ~1,00 é ilusória pelo desbalanceamento e **não** deve ser usada como headline.
- **Geração de Predições:** Previsões consolidadas em `predictions_classes.csv`.
- **Organização:** Código reestruturado em scripts Python em `Projeto/src/` (`data_preparation`, `exploratory_analysis`, `model_training`, `prediction_generation`, `utils`, `main`).
