import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE


def mapear_erros(df, colunas_para_mapear):
 
    mapeamento_falhas = {
        'não': 0,
        'Não': 0,
        'sim': 1,
        'N': 0,
        'Sim': 1,
        'y': 1,
        '1': 1,
        '0': 0,
        False: 0,
        True: 1,
        'False': 0,
        'True': 1,
        'n': 0,
        'nao': 0,
        's': 1,
        '-': 0, 
    }

    for coluna in colunas_para_mapear:
        # Verifica se a coluna existe antes de tentar o mapeamento
        if coluna in df.columns:
            # Aplica o mapeamento e trata possíveis NaN (que podem surgir de valores não mapeados)
            df[coluna] = df[coluna].map(mapeamento_falhas)

            # Converte a coluna para tipo numérico, forçando erros para NaN
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
        else:
            print(f"Aviso: A coluna '{coluna}' não foi encontrada no DataFrame.")

    return df


def train_and_evaluate_model(df, target_column, model_type, param_distributions=None, n_iter_search=10):
    """
    Args:
        df (pd.DataFrame): O DataFrame de entrada com features e coluna alvo.
        target_column (str): O nome da coluna alvo para treinamento.
        model_type (str): O tipo de modelo a ser usado ('RandomForest' ou 'GradientBoosting').
        param_distributions (dict, optional): Distribuição de parâmetros para RandomizedSearchCV.
                                              Padrão é None (sem ajuste de hiperparâmetros).
        n_iter_search (int, optional): Número de iterações para RandomizedSearchCV. Padrão é 10.

    Returns:
        tuple: Uma tupla contendo:
            - best_pipeline (Pipeline): O pipeline treinado (ou o melhor estimador do RandomizedSearchCV).
            - report (dict): O classification_report do conjunto de teste.
            - y_pred (np.ndarray): As previsões no conjunto de teste.
    """
    # 1. Separar features (X) e target (y)
    # Remove colunas de ID e todas as colunas de falha (exceto a target_column atual) das features
    falha_columns = ['falha_maquina', 'FDF (Falha Desgaste Ferramenta)',
                     'FDC (Falha Dissipacao Calor)', 'FP (Falha Potencia)',
                     'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)']
    cols_to_drop = ['id', 'id_produto'] + [col for col in falha_columns if col != target_column]

    y = df[target_column].copy()
    X = df.drop(columns=cols_to_drop, errors='ignore').copy() # Use errors='ignore' para evitar erro se a coluna já foi dropada

    # 2. Separar dados em treino e teste
    # Usar stratify para garantir proporções de classes semelhantes em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Identificar colunas numéricas e categóricas em X_train
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

    # 4. Identificar colunas numéricas com valores nulos em X_train para imputação
    # Filtra apenas colunas numéricas que realmente contêm nulos no conjunto de treino
    numerical_cols_with_missing = [col for col in numerical_cols if X_train[col].isnull().any()]
    numerical_cols_no_missing = [col for col in numerical_cols if col not in numerical_cols_with_missing]


    # 5. Definir o pré-processamento usando ColumnTransformer
    # Inclui imputação para colunas numéricas com nulos antes do escalonamento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_impute_scale', Pipeline([
                ('imputador', SimpleImputer(strategy='median')), # Imputa nulos com a mediana
                ('escalador', MinMaxScaler()) # Escalona os dados
            ]), numerical_cols_with_missing),
            ('num_scale', MinMaxScaler(), numerical_cols_no_missing), # Escalona colunas numéricas sem nulos
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols) # Aplica OneHotEncoder em colunas categóricas
        ],
        remainder='passthrough' # Mantém outras colunas (se houver) sem transformação
    )

    # Aplicar pré-processamento nos conjuntos de treino e teste
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Aplicar SMOTE apenas para a coluna 'FA (Falha Aleatoria)' para lidar com desbalanceamento de classe
    if target_column == 'FA (Falha Aleatoria)':
        print(f"Aplicando SMOTE para balancear a classe minoritária em {target_column}...")
        smote = SMOTE(random_state=42)
        # Aplicar SMOTE nos dados de treino pré-processados
        X_train_final, y_train_final = smote.fit_resample(X_train_processed, y_train)
        print(f"Forma dos dados de treino após SMOTE: {X_train_final.shape}")
    else:
        # Se não for 'FA (Falha Aleatoria)', usar os dados pré-processados diretamente
        X_train_final = X_train_processed
        y_train_final = y_train


    # 6. Criar o pipeline do modelo com o classificador
    # O pipeline agora contém apenas o classificador, pois o pré-processamento é feito antes do SMOTE
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
    else:
        print(f"Tipo de modelo inválido: {model_type}. Escolha 'RandomForest' ou 'GradientBoosting'.")
        return None, None, None

    model_pipeline = Pipeline(steps=[
        ('classifier', model)
    ])

    # 7. Executar RandomizedSearchCV se distribuições de parâmetros forem fornecidas
    if param_distributions:
        print(f"Executando RandomizedSearchCV para {target_column} com {model_type}...")
        random_search = RandomizedSearchCV(
            model_pipeline, # O pipeline agora contém apenas o classificador
            param_distributions=param_distributions,
            n_iter=n_iter_search, # Número de combinações de parâmetros a testar
            cv=5, # Validação cruzada de 5 folds
            scoring='f1', # Usando F1-score como métrica de avaliação, adequada para classes desbalanceadas
            random_state=42,
            n_jobs=-1 # Usar todos os núcleos disponíveis para paralelizar
        )
        # Ajustar RandomizedSearchCV nos dados de treino (possivelmente reamostrados pelo SMOTE)
        random_search.fit(X_train_final, y_train_final)

        print("Melhores parâmetros encontrados:")
        print(random_search.best_params_)
        print(f"Melhor score F1 de validação cruzada: {random_search.best_score_:.4f}")

        # Usar o melhor estimador (pipeline com os melhores parâmetros) para predição e avaliação
        best_pipeline = random_search.best_estimator_
    else:
        print(f"Treinando {model_type} para {target_column} sem ajuste de hiperparâmetros...")
        # Ajustar o pipeline (apenas o classificador) nos dados de treino (possivelmente reamostrados pelo SMOTE)
        model_pipeline.fit(X_train_final, y_train_final)
        best_pipeline = model_pipeline


    # 8. Fazer previsões e avaliar o modelo
    # Predizer no conjunto de teste original (não reamostrado pelo SMOTE)
    y_pred = best_pipeline.predict(X_test_processed)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nRelatório de Classificação no Conjunto de Teste:")
    print(classification_report(y_test, y_pred))

    # Gerar e exibir a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]) # Definir labels explicitamente
    disp.plot(cmap='Blues')
    plt.title(f"Matriz de Confusão para '{target_column}' - {model_type} (Ajustado)" if param_distributions else f"Matriz de Confusão para '{target_column}' - {model_type}")
    plt.tight_layout()
    plt.show()

    # Retornar o melhor pipeline, relatório e previsões no conjunto de teste
    return best_pipeline, report, y_pred
