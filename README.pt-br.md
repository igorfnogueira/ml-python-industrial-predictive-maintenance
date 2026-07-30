
Language / Idioma: [English](README.md) | **Português**

# Relatório de Análise e Modelagem para Manutenção Preditiva

Este projeto teve como objetivo desenvolver e implementar modelos de manutenção preditiva para identificar diversos tipos de falhas em máquinas. A metodologia incluiu preparação de dados, análise exploratória e modelagem multi-rótulo.

## 1. Preparação e Limpeza dos Dados

- **Identificação e Tratamento de Inconsistências:** Valores de falha com diferentes grafias (e.g., 'sim', 'Sim', 'True', 'não', 'Não', 'False', 'N', 'y', '1', '0') foram padronizados para 1 (falha) ou 0 (sem falha). Valores anômalos como '-' foram mapeados para 0, assumindo ausência de falha.

- **Tratamento de Nulos e Anomalias Físicas:** Valores negativos nas colunas de temperatura (em Kelvin) foram removidos e substituídos por `NaN` para posterior imputação, pois são fisicamente impossíveis. Após a padronização, as colunas binárias de falha foram convertidas para o tipo inteiro. Os valores nulos resultantes da limpeza e os existentes foram tratados com imputação da mediana dentro do pipeline de pré-processamento.

- **Gestão de Outliers e Sinais Operacionais:** Outliers e valores negativos em colunas como `velocidade_rotacional` e `desgaste_da_ferramenta` foram mantidos. Testes mostraram que a inclusão desses valores resultou em modelos com desempenho superior, indicando que representam sinais operacionais importantes, e não ruído. Para a `umidade_relativa`, apesar dos outliers, o baixo desvio padrão e a coincidência de média, moda e mediana sugerem que esses "outliers" são variações válidas e foram mantidos.

## 2. Análise Exploratória e Visualização

- **Estatísticas Descritivas:** Calculadas para as colunas numéricas (média, mediana, moda, desvio padrão) para entender a distribuição dos dados.

- **Análise de Relações:** Boxplots foram utilizados para visualizar a relação entre atributos dos sensores e as falhas. Por exemplo, `desgaste_da_ferramenta` foi um forte preditor para `FDF (Falha Desgaste Ferramenta)`, e `torque` e `desgaste_da_ferramenta` foram cruciais para `FTE (Falha Tensao Excessiva)`.

## 3. Metodologia de Modelagem e Treinamento

- **Estratégia de Classificação:** Abordagem de classificação multi-rótulo, com 5 modelos treinados independentemente, um para cada tipo de falha.

- **Pipeline de Pré-processamento:** Um `ColumnTransformer` foi empregado para:
    - Imputar valores nulos em colunas numéricas com a mediana.
    - Escalonar colunas numéricas com `MinMaxScaler`.
    - Aplicar `OneHotEncoder` em colunas categóricas.

- **Seleção de Modelos:** `RandomForestClassifier` e `GradientBoostingClassifier` foram escolhidos devido à sua alta precisão, capacidade de fornecer insights sobre a importância das features e robustez com dados tabulares. O modelo com melhor desempenho para cada falha específica foi selecionado.

- **Otimização de Hiperparâmetros:** `RandomizedSearchCV` foi usado com validação cruzada (5 folds) e `f1-score` como métrica para otimizar os hiperparâmetros dos modelos, buscando a melhor performance.

- **Otimização para Classes Desbalanceadas:** Para `FA (Falha Aleatoria)`, o `SMOTE` foi aplicado para rebalancear o conjunto de treino. Embora aplicado, o modelo para `FA` ainda apresentou desafios para obter bons resultados.

## 4. Resultados e Conclusão

- **Desempenho dos Modelos:**
    - `RandomForest`: Melhor desempenho para `FDF (Falha Desgaste Ferramenta)`, `FP (Falha Potencia)` e `FA (Falha Aleatoria)`.
    - `Gradient Boosting`: Foi o melhor desempenho para `FDC (Falha Dissipacao Calor)` e `FTE (Falha Tensao Excessiva)`.

- **Validação da Hipótese de Outliers:** A decisão de manter outliers e valores negativos em `velocidade_rotacional` e `desgaste_da_ferramenta` foi validada pelos resultados, que foram superiores aos modelos treinados sem eles.

- **Geração de Predições:** As previsões dos 5 modelos foram consolidadas em um arquivo CSV (`predictions_classes.csv`).

- **Organização do Projeto:** O código foi reestruturado em scripts Python com documentação para facilitar a organização e manutenção.
