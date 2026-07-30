
Language / Idioma: **English** | [Português](README.pt-br.md)

# Predictive Maintenance Analysis and Modeling Report

This project aimed to develop and implement predictive maintenance models to identify various types of machine failures. The methodology included data preparation, exploratory analysis, and multi-label modeling.

## 1. Data Preparation and Cleaning

- **Identification and Treatment of Inconsistencies:** Failure values with different spellings (e.g., 'sim', 'Sim', 'True', 'não', 'Não', 'False', 'N', 'y', '1', '0') were standardized to 1 (failure) or 0 (no failure). Anomalous values like '-' were mapped to 0, assuming no failure.

- **Handling Nulls and Physical Anomalies:** Negative values in temperature columns (in Kelvin) were removed and replaced with `NaN` for later imputation, as they are physically impossible. After standardization, binary failure columns were converted to integer type. Null values resulting from cleaning and existing ones were handled with median imputation within the preprocessing pipeline.

- **Outlier and Operational Signal Management:** Outliers and negative values in columns such as `velocidade_rotacional` (rotational speed) and `desgaste_da_ferramenta` (tool wear) were retained. Tests showed that including these values resulted in models with superior performance, indicating that they represent important operational signals rather than noise. For `umidade_relativa` (relative humidity), despite the presence of outliers, the low standard deviation and the coincidence of mean, mode, and median suggest that these "outliers" are valid variations and were therefore kept.

## 2. Exploratory Analysis and Visualization

- **Descriptive Statistics:** Calculated for numerical columns (mean, median, mode, standard deviation) to understand data distribution.

- **Relationship Analysis:** Boxplots were used to visualize the relationship between sensor attributes and failures. For instance, `desgaste_da_ferramenta` was a strong predictor for `FDF (Falha Desgaste Ferramenta)` (Tool Wear Failure), and `torque` and `desgaste_da_ferramenta` were crucial for `FTE (Falha Tensao Excessiva)` (Overvoltage Failure).

## 3. Modeling and Training Methodology

- **Classification Strategy:** A multi-label classification approach was adopted, with 5 models trained independently, one for each failure type.

- **Preprocessing Pipeline:** A `ColumnTransformer` was used to:
    - Impute null values in numerical columns with the median.
    - Scale numerical columns using `MinMaxScaler`.
    - Apply `OneHotEncoder` to categorical columns.

- **Model Selection:** `RandomForestClassifier` and `GradientBoostingClassifier` were chosen due to their high accuracy, ability to provide feature importance insights, and robustness with tabular data. The best-performing model for each specific failure was selected.

- **Hyperparameter Optimization:** `RandomizedSearchCV` was employed with 5-fold cross-validation and `f1-score` as the metric to optimize model hyperparameters, aiming for the best performance.

- **Optimization for Imbalanced Classes:** For `FA (Falha Aleatoria)` (Random Failure), `SMOTE` was applied to rebalance the training set. Despite its application, the model for `FA` still presented challenges in achieving good results.

## 4. Results and Conclusion

- **Model Performance:**
    - `RandomForest`: Achieved the best performance for `FDF (Falha Desgaste Ferramenta)`, `FP (Falha Potencia)` (Power Failure), and `FA (Falha Aleatoria)`.
    - `Gradient Boosting`: Was the best-performing model for `FDC (Falha Dissipacao Calor)` (Heat Dissipation Failure) and `FTE (Falha Tensao Excessiva)`.

- **Outlier Hypothesis Validation:** The decision to retain outliers and negative values in `velocidade_rotacional` and `desgaste_da_ferramenta` was validated by the results, which were superior to models trained without them.

- **Prediction Generation:** Predictions from the 5 models were consolidated into a CSV file (`predictions_classes.csv`).

- **Project Organization:** The code was restructured into Python scripts with appropriate documentation for easy organization and maintenance.
