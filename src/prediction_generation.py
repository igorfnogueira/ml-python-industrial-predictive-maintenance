import pandas as pd
import joblib
from data_preparation import preparar_dados
from sklearn.model_selection import train_test_split
from config import DATA_RAW_CSV, DEFAULT_PREDICTIONS_CSV, model_path

def gerar_predicoes_csv(df_processado, pipelines_treinados, caminho_saida_csv=None):
    """
    Gera previsões para cada tipo de falha usando os pipelines treinados
    e salva as previsões em um arquivo CSV.

    Args:
        df_processado (pd.DataFrame): O DataFrame processado contendo as features.
        pipelines_treinados (dict): Um dicionário contendo os pipelines treinados para cada tipo de falha.
        caminho_saida_csv (str, optional): O caminho e nome do arquivo CSV de saída.
                                         Padrão é outputs/predictions_classes.csv.
    """
    if caminho_saida_csv is None:
        caminho_saida_csv = str(DEFAULT_PREDICTIONS_CSV)

    print("Iniciando a geração de predições e salvamento em CSV...")

    # Define as colunas alvo (tipos de falha) que foram usadas para treinamento
    failure_columns = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)',
                       'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)',
                       'FA (Falha Aleatoria)']

   

    # Seleciona as features para a divisão (excluindo colunas de ID e todas as falhas)
    cols_para_divisao = [col for col in df_processado.columns if col not in ['id', 'id_produto'] + failure_columns]
    X_dummy = df_processado[cols_para_divisao].copy()
    # Usamos uma coluna dummy para o 'y' apenas para estratificar a divisão
    y_dummy = df_processado[failure_columns[0]].copy() # Pode ser qualquer coluna de falha

    # Realiza a divisão treino/teste para obter os índices do conjunto de teste
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
        X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy
    )

    # Obtém os índices do conjunto de teste do dataframe processado
    test_indices = X_test_dummy.index

    df_test_original = df_processado.loc[test_indices].copy()

    # Cria um dicionário para armazenar as previsões
    predictions_data = {'id': df_test_original['id']}

    
    y_dummy = df_processado['FA (Falha Aleatoria)'].copy() # Usando uma das colunas alvo apenas para obter os índices da divisão
    # Seleciona as features usadas no treinamento (excluindo IDs e todas as falhas)
    features_para_divisao = [col for col in df_processado.columns if col not in ['id', 'id_produto', 'falha_maquina',
                                      'FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)',
                                      'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)',
                                      'FA (Falha Aleatoria)']]

    X_dummy = df_processado[features_para_divisao].copy()


    # Realiza a divisão treino/teste 
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
        X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy
    )

    # Obtém os índices do conjunto de teste do dataframe processado
    test_indices = X_test_dummy.index


    df_test_com_ids = df_processado.loc[test_indices].copy()

    # Carregar as previsões dos arquivos .pkl (simulação para script standalone)
    predictions_dict = {}
    for falha in failure_columns:
        model_filename = model_path(falha)
        try:
            # Carregar o pipeline treinado
            pipeline = joblib.load(model_filename)

            # Carregar o dataframe processado (que já tem o tratamento inicial e NaNs nas temperaturas negativas)
            df_processado = preparar_dados(str(DATA_RAW_CSV))

            # Definir as colunas alvo (tipos de falha)
            failure_columns = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)',
                               'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)',
                               'FA (Falha Aleatoria)']

            # Selecionar features e target para a divisão treino/teste
            # Remove colunas de ID e todas as colunas de falha das features
            cols_to_drop_for_features = ['id', 'id_produto'] + failure_columns
            X = df_processado.drop(columns=cols_to_drop_for_features, errors='ignore').copy()
            
            y_dummy = df_processado[failure_columns[0]].copy() # Pode ser qualquer coluna de falha

            # Realiza a divisão treino/testete
            X_train, X_test, y_train_dummy, y_test_dummy = train_test_split(
                X, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy
            )

            # Obtém os índices do conjunto de teste
            test_indices = X_test.index

            

            # Identificar colunas numéricas e categóricas em X_train para o preprocessor
            numerical_cols_train = X_train.select_dtypes(include=np.number).columns.tolist()
            categorical_cols_train = X_train.select_dtypes(include='object').columns.tolist()

            # Identificar colunas numéricas com valores nulos em X_train para imputação
            numerical_cols_with_missing_train = [col for col in numerical_cols_train if X_train[col].isnull().any()]
            numerical_cols_no_missing_train = [col for col in numerical_cols_train if col not in numerical_cols_with_missing_train]

            # Definir e fitar o pré-processamento no conjunto de treino
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_impute_scale', Pipeline([
                        ('imputador', SimpleImputer(strategy='median')),
                        ('escalador', MinMaxScaler())
                    ]), numerical_cols_with_missing_train),
                    ('num_scale', MinMaxScaler(), numerical_cols_no_missing_train),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_train)
                ],
                remainder='passthrough'
            )

            # Fitar o preprocessor nos dados de treino
            preprocessor.fit(X_train)

            # Transformar o conjunto de teste usando o preprocessor fitado
            X_test_processed = preprocessor.transform(X_test)

            # Carregar os pipelines treinados e gerar previsões no X_test_processed
            predictions_dict = {}
            print("\nCarregando modelos e gerando previsões no conjunto de teste...")
            for falha in failure_columns:
                model_filename = model_path(falha)
                try:
                    pipeline = joblib.load(model_filename)
                    # Fazer previsões no conjunto de teste pré-processado
                    y_pred = pipeline.predict(X_test_processed)
                    predictions_dict[falha] = y_pred
                    print(f"Previsões geradas para '{falha}'.")
                except FileNotFoundError:
                    print(f"Erro: Arquivo do modelo '{model_filename}' não encontrado. Certifique-se de que os modelos foram treinados e salvos.")
                    return None # Retorna None se um modelo não for encontrado
                except Exception as e:
                    print(f"Erro ao carregar o modelo ou gerar previsões para '{falha}': {e}")
                    return None


            # Seleciona os IDs do conjunto de teste do DataFrame processado (que tem os IDs originais)
            df_test_ids = df_processado.loc[test_indices, ['id']].copy()

            # Cria um dicionário com os IDs e as previsões
            predictions_data = {'id': df_test_ids['id']}
            for falha, preds in predictions_dict.items():
                predictions_data[falha] = preds

            # Cria o DataFrame de previsões
            predictions_df = pd.DataFrame(predictions_data)

            # Salva o DataFrame em um arquivo CSV
            predictions_df.to_csv(caminho_saida_csv, index=False)

            print(f"\nArquivo CSV com classes preditas gerado com sucesso: {caminho_saida_csv}")
            print("\nPrimeiras 10 linhas do arquivo de previsões:")
            print(predictions_df.head(10))

            return predictions_df


if __name__ == '__main__':
    caminho_dados_original = str(DATA_RAW_CSV)
    caminho_saida = str(DEFAULT_PREDICTIONS_CSV.with_name('predictions_classes_standalone.csv'))

    df_processado = preparar_dados(caminho_dados_original)

    predictions_df = gerar_predicoes_csv(df_processado, pipelines_treinados=None, caminho_saida_csv=caminho_saida)
