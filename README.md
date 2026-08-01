Language / Idioma: **English** | [Português](README.pt-br.md)

# Predictive Maintenance Analysis and Modeling Report

This project builds predictive maintenance models to identify five machine failure types from sensor data, covering data preparation, exploratory analysis, and multi-label classification (one binary classifier per failure).

**Dataset:** `data/raw/bootcamp_train.csv` — 35,260 rows × 15 columns (AI4I-like schema: `tipo` L/M/H, temperatures, torque, tool wear, etc., plus `umidade_relativa`). A formal source citation (UCI/Kaggle/plant data) is **not** documented in this repository.

## Repository structure

```
.
├── data/raw/                 # Input dataset
├── notebooks/                # Official experiment (Colab/bootcamp)
├── src/                      # Python pipeline scripts
│   ├── config.py             # Paths relative to repo root
│   ├── data_preparation.py
│   ├── exploratory_analysis.py
│   ├── model_training.py
│   ├── prediction_generation.py
│   ├── utils.py
│   └── main.py
├── models/                   # .pkl artifacts (generated)
├── outputs/                  # Predictions and outputs (generated)
├── requirements.txt
├── README.md
└── README.pt-br.md
```

Run from `src/`:

```bash
pip install -r requirements.txt
cd src
python main.py
```

## 1. Data Preparation and Cleaning

- **Identification and Treatment of Inconsistencies:** Failure values with different spellings (e.g., 'sim', 'Sim', 'True', 'não', 'Não', 'False', 'N', 'y', '1', '0') were standardized to 1 (failure) or 0 (no failure). Anomalous values like '-' were mapped to 0, assuming no failure.

- **Handling Nulls and Physical Anomalies:** Negative values in temperature columns (in Kelvin) were replaced with `NaN` for later imputation, as they are physically impossible. After standardization, binary failure columns were converted to integer type. Nulls in FDF/FA were filled with 0; remaining numeric nulls were handled with median imputation in the preprocessing pipeline.

- **Outlier and Operational Signal Management:** Outliers and negative values in `velocidade_rotacional` (rotational speed) and `desgaste_da_ferramenta` (tool wear) were **kept** under the hypothesis that they are operational signal, not sensor noise. For `umidade_relativa` (relative humidity), matching mean/mode/median and low standard deviation also supported keeping extreme points. A numeric A/B comparison (with vs without outliers) is **not versioned** in this repository — the decision remains a domain hypothesis.

## 2. Exploratory Analysis and Visualization

- **Descriptive Statistics:** Computed for numeric columns (mean, median, mode, standard deviation).

- **Relationship Analysis:** Boxplots relating sensor attributes to failures. Examples: `desgaste_da_ferramenta` as a strong predictor of `FDF`; `torque` and `desgaste_da_ferramenta` relevant for `FTE`.

## 3. Modeling and Training Methodology

- **Classification Strategy:** Multi-label setup via 5 independent binary classifiers (one failure type per model).

- **Preprocessing Pipeline:** `ColumnTransformer` with median imputation, `MinMaxScaler`, and `OneHotEncoder` for categorical columns.

- **Models used in the official experiment** (`notebooks/Projeto_Final_do_Bootcamp_CDIA.ipynb`):
  - `RandomForestClassifier`: FDF, FP, FA
  - `GradientBoostingClassifier`: FDC, FTE  
  A formal RF vs GB head-to-head table per failure is **not versioned** in the notebook/scripts; the assignment above is what was executed and reported.

- **Evaluation:** Stratified 80/20 holdout (`test_size=0.2`, `stratify=y`, `random_state=42`). Headline metric = **positive-class F1** on the **test** set.

- **Hyperparameter Optimization:** `RandomizedSearchCV` on training data only (`cv=5`, `scoring='f1'`, `n_iter=10`, `random_state=42`). Single run with fixed seed; full trial table was not published.

- **Imbalanced classes:** `SMOTE` applied **only on training data** and **only** for `FA (Falha Aleatoria)` (Random Failure). The test set is never resampled.

- **Feature caveat (official notebook):** In the notebook's `train_and_evaluate_model`, building `X` drops `id`, `id_produto`, and the current target, but **keeps** `falha_maquina` and the other failure-type columns — a multi-label *label leakage* risk. Scripts under `src/` use a slightly different drop logic; the metrics below come from the notebook.

## 4. Results and Conclusion

Official metrics on the **test holdout** (7,052 samples), positive-class F1:

| Failure | Model | F1 (class 1) | Positives in test | CV F1 (train) |
|---|---|---|---|---|
| FDF (Tool Wear) | Random Forest | **0.67** | 14 | 0.56 |
| FDC (Heat Dissipation) | Gradient Boosting | **0.88** | 45 | 0.90 |
| FP (Power) | Random Forest | **0.71** | 25 | 0.78 |
| FTE (Overstrain / Overvoltage) | Gradient Boosting | **0.89** | 34 | 0.74 |
| FA (Random) | Random Forest + SMOTE | **0.00** | 15 | 0.996* |

\*FA CV score is on SMOTE-augmented training folds — optimistic vs test F1 = 0.00. The 15/7,052 counts are from the **test** set, not full-dataset prevalence (~order of 75/35,260 for FA).

- **Summary:** Positive-class F1 from **0.67 to 0.89** on 4 of 5 failures (best: FTE = 0.89). FA did not converge (extremely rare class). Accuracy ~1.00 is misleading under imbalance and should **not** be used as a headline.
- **Prediction Generation:** Predictions consolidated into `outputs/predictions_classes.csv`.
- **Project Organization:** Code lives under `src/` with paths centralized in `config.py`; models in `models/` and outputs in `outputs/`.
