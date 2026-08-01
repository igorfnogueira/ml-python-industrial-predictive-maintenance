import pandas as pd
import numpy as np
from utils import mapear_erros
from config import DATA_RAW_CSV

def preparar_dados(caminho_arquivo_csv):
    """
    Carrega, limpa e realiza o pré-processamento inicial dos dados de manutenção preditiva.

    Args:
        caminho_arquivo_csv (str): O caminho para o arquivo CSV contendo os dados.

    Returns:
        pd.DataFrame: O DataFrame processado e limpo.
    """
    print(f"Carregando dados do arquivo: {caminho_arquivo_csv}")
    # Carrega o dataset a partir do arquivo CSV
    df_original = pd.read_csv(caminho_arquivo_csv)

    # Cria uma cópia do DataFrame original para trabalhar
    df_processado = df_original.copy()
    print("Cópia do DataFrame original criada.")

    # Define as colunas de falha para mapeamento
    colunas_para_mapear = ['falha_maquina', 'FDF (Falha Desgaste Ferramenta)',
                           'FDC (Falha Dissipacao Calor)', 'FP (Falha Potencia)',
                           'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)']

    # Mapeia diferentes representações de falha para 0 ou 1
    print("Mapeando representações de falha para 0 ou 1...")
    df_processado = mapear_erros(df_processado, colunas_para_mapear)
    print("Mapeamento de falhas concluído.")

    # Trata valores negativos nas colunas de temperatura (fisicamente impossíveis em Kelvin)
    print("Tratando valores negativos nas colunas de temperatura...")
    df_processado.loc[df_processado['temperatura_ar'] < 0, 'temperatura_ar'] = np.nan
    df_processado.loc[df_processado['temperatura_processo'] < 0, 'temperatura_processo'] = np.nan
    print("Tratamento de valores negativos de temperatura concluído.")

    # Trata os valores nulos nas colunas específicas de falha, substituindo por 0
    print("Tratando valores nulos em colunas de falha (substituindo por 0)...")
    df_processado['FDF (Falha Desgaste Ferramenta)'] = df_processado['FDF (Falha Desgaste Ferramenta)'].fillna(0)
    df_processado['FA (Falha Aleatoria)'] = df_processado['FA (Falha Aleatoria)'].fillna(0)
    print("Tratamento de valores nulos em colunas de falha concluído.")

    # Converte as colunas de falha para o tipo inteiro
    print("Convertendo colunas de falha para tipo inteiro...")
    df_processado['FDF (Falha Desgaste Ferramenta)'] = df_processado['FDF (Falha Desgaste Ferramenta)'].astype(int)
    df_processado['FA (Falha Aleatoria)'] = df_processado['FA (Falha Aleatoria)'].astype(int)
    # Converte outras colunas de falha que já foram mapeadas para int
    df_processado['falha_maquina'] = df_processado['falha_maquina'].astype(int)
    df_processado['FDC (Falha Dissipacao Calor)'] = df_processado['FDC (Falha Dissipacao Calor)'].astype(int)
    df_processado['FP (Falha Potencia)'] = df_processado['FP (Falha Potencia)'].astype(int)
    df_processado['FTE (Falha Tensao Excessiva)'] = df_processado['FTE (Falha Tensao Excessiva)'].astype(int)
    print("Conversão de tipo das colunas de falha concluída.")

    print("Pré-processamento inicial dos dados concluído.")
    return df_processado

if __name__ == '__main__':
    caminho_dados = str(DATA_RAW_CSV)
    df_processado = preparar_dados(caminho_dados)

    print("\nPrimeiras 5 linhas do DataFrame processado:")
    print(df_processado.head())

    print("\nVerificando valores nulos após o pré-processamento inicial:")
    print(df_processado.isnull().sum())

    print("\nVerificando os tipos de dados após o pré-processamento inicial:")
    print(df_processado.dtypes)
