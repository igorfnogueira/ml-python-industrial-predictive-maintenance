import pandas as pd
from utils import train_and_evaluate_model
from data_preparation import preparar_dados
from scipy.stats import randint, uniform
import joblib
from config import DATA_RAW_CSV, MODELS_DIR, model_path

def treinar_modelos(df):
    """
    Treina modelos de classificação para cada tipo de falha utilizando
    RandomForest e GradientBoosting com ajuste de hiperparâmetros.

    Args:
        df (pd.DataFrame): O DataFrame processado contendo as features e colunas alvo.

    Returns:
        dict: Um dicionário contendo os pipelines treinados para cada tipo de falha.
        dict: Um dicionário contendo os relatórios de classificação para cada tipo de falha.
        dict: Um dicionário contendo as previsões para cada tipo de falha no conjunto de teste.
    """
    print("Iniciando treinamento dos modelos para cada tipo de falha...")

    # Define as colunas alvo (tipos de falha)
    failure_columns = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)',
                       'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)',
                       'FA (Falha Aleatoria)']

    # Define as distribuições de parâmetros para RandomizedSearchCV
    rf_param_dist = {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': [None, 10, 20, 30, 40, 50],
        'classifier__min_samples_split': randint(2, 20),
        'classifier__min_samples_leaf': randint(1, 20),
        'classifier__bootstrap': [True, False]
    }

    gb_param_dist = {
        'classifier__n_estimators': randint(100, 500),
        'classifier__learning_rate': uniform(0.01, 0.2),
        'classifier__max_depth': randint(3, 10),
        'classifier__min_samples_split': randint(2, 20),
        'classifier__min_samples_leaf': randint(1, 20),
        'classifier__subsample': uniform(0.6, 0.4)
    }

    # Dicionários para armazenar os pipelines, relatórios e previsões
    pipelines = {}
    reports = {}
    predictions = {}

    # Treina e avalia um modelo para cada tipo de falha
    for target_failure in failure_columns:
        print(f"\nTreinando modelo para: {target_failure}")

        # Escolhe o tipo de modelo e distribuição de parâmetros
        if target_failure in ['FDF (Falha Desgaste Ferramenta)', 'FP (Falha Potencia)', 'FA (Falha Aleatoria)']:
             model_type = 'RandomForest'
             param_dist = rf_param_dist
        elif target_failure in ['FDC (Falha Dissipacao Calor)', 'FTE (Falha Tensao Excessiva)']:
             model_type = 'GradientBoosting'
             param_dist = gb_param_dist
        else:
             print(f"Tipo de falha '{target_failure}' não reconhecido. Pulando treinamento.")
             continue

        # Chama a função train_and_evaluate_model do módulo utils
        pipeline, report, y_pred = train_and_evaluate_model(
            df,
            target_failure,
            model_type,
            param_distributions=param_dist,
            n_iter_search=10 # Define o número de iterações para RandomizedSearchCV
        )

        # Armazena o pipeline treinado, relatório e previsões
        if pipeline:
            pipelines[target_failure] = pipeline
            reports[target_failure] = report
            predictions[target_failure] = y_pred

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_filename = model_path(target_failure)
            joblib.dump(pipeline, model_filename)
            print(f"Modelo treinado para '{target_failure}' salvo como '{model_filename}'")


    print("\nTreinamento de modelos concluído.")
    return pipelines, reports, predictions

if __name__ == '__main__':
    caminho_dados = str(DATA_RAW_CSV)
    df_processado = preparar_dados(caminho_dados)

    # Treina os modelos
    pipelines_treinados, relatorios, predicoes = treinar_modelos(df_processado)

    # Exibe um resumo dos relatórios de classificação
    print("\nResumo dos Relatórios de Classificação:")
    for falha, relatorio in relatorios.items():
        print(f"\nRelatório para '{falha}':")
        # Imprime o relatório formatado
        print(f"  Acurácia: {relatorio['accuracy']:.4f}")
        print(f"  F1-score (macro avg): {relatorio['macro avg']['f1-score']:.4f}")
        print(f"  F1-score (weighted avg): {relatorio['weighted avg']['f1-score']:.4f}")

    # Os pipelines treinados e as previsões estão disponíveis nos dicionários 'pipelines_treinados' e 'predicoes'
