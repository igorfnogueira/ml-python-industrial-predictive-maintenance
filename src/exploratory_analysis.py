import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_preparation import preparar_dados
from config import DATA_RAW_CSV

def realizar_analise_exploratoria(df):
    """
    Realiza a análise exploratória e visualização dos dados operacionais e de falha.

    Inclui estatísticas descritivas, histogramas e boxplots para variáveis numéricas,
    e boxplots comparando variáveis operacionais com colunas de falha.

    Args:
        df (pd.DataFrame): O DataFrame processado para análise.
    """
    print("Iniciando análise exploratória dos dados...")

    # Define os atributos operacionais e as colunas de falha
    operational_features = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa',
                            'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
    failure_columns = ['FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)',
                       'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)',
                       'FA (Falha Aleatoria)']

    print("\nCalculando estatísticas descritivas para features operacionais:")
    for feature in operational_features:
        if feature in df.columns:
            media = df[feature].mean()
            moda = df[feature].mode()
            mediana = df[feature].median()
            desvio_padrao = df[feature].std()

            print(f"\nEstatísticas para '{feature}':")
            print(f"  Média: {media:.2f}")
            # Verifica se a moda não está vazia antes de imprimir
            if not moda.empty:
                 print(f"  Moda: {moda.iloc[0]:.2f}" if pd.api.types.is_numeric_dtype(moda) else f"  Moda: {moda.iloc[0]}")
            else:
                 print("  Moda: Não aplicável (nenhum valor único ou múltiplos valores com a mesma alta frequência)")

            print(f"  Mediana: {mediana:.2f}")
            print(f"  Desvio Padrão: {desvio_padrao:.2f}")
        else:
            print(f"Aviso: A coluna '{feature}' não foi encontrada no DataFrame.")


    print("\nGerando histogramas e boxplots para features operacionais...")
    # Criando gráficos de distribuição (histograma e boxplot) para features operacionais
    for feature in operational_features:
         if feature in df.columns:
            plt.figure(figsize=(14, 6))

            # Histograma
            plt.subplot(1, 2, 1)
            sns.histplot(df[feature].dropna(), bins=30, kde=True, color="skyblue") # Usa dropna para ignorar NaNs nos gráficos
            plt.title(f"Distribuição de {feature}")
            plt.xlabel(feature.replace('_', ' ').title()) # Formata o nome da feature para o título do eixo
            plt.ylabel("Frequência")

            # Boxplot com estatísticas
            plt.subplot(1, 2, 2)
            sns.boxplot(y=df[feature].dropna(), color="lightcoral") # Usa dropna para ignorar NaNs nos gráficos
            plt.title(f"Boxplot de {feature}")
            plt.ylabel(feature.replace('_', ' ').title())

            # Adicionando linhas para média, mediana e +/- 1 desvio padrão
            media = df[feature].mean()
            mediana = df[feature].median()
            desvio_padrao = df[feature].std()
            moda_vals = df[feature].mode() # Pode retornar múltiplos valores

            if not pd.isna(media):
                plt.axhline(media, color='green', linestyle='--', label=f'Média: {media:.2f}')
            if not pd.isna(mediana):
                plt.axhline(mediana, color='red', linestyle='-', label=f'Mediana: {mediana:.2f}')
            if not moda_vals.empty and not pd.isna(moda_vals.iloc[0]):
                # Adiciona apenas a primeira moda se houver valores de moda e o primeiro não for NaN
                 plt.axhline(moda_vals.iloc[0], color='purple', linestyle=':', label=f'Moda: {moda_vals.iloc[0]:.2f}' if pd.api.types.is_numeric_dtype(moda_vals) else f'Moda: {moda_vals.iloc[0]}')

            # Adiciona linhas para +/- 1 desvio padrão apenas se desvio_padrao não for NaN
            if not pd.isna(desvio_padrao):
                if not pd.isna(media + desvio_padrao):
                    plt.axhline(media + desvio_padrao, color='orange', linestyle='-.', label=f'+1σ: {media + desvio_padrao:.2f}')
                if not pd.isna(media - desvio_padrao):
                     plt.axhline(media - desvio_padrao, color='orange', linestyle='-.', label=f'-1σ: {media - desvio_padrao:.2f}')


            plt.legend()

            plt.tight_layout() # Ajusta o layout para evitar sobreposição
            plt.show()
         else:
             print(f"Aviso: Coluna '{feature}' não encontrada para plotagem.")


    print("\nGerando boxplots comparando features operacionais com colunas de falha...")
    # Criando box plots para cada atributo operacional vs cada tipo de falha
    for target_failure in failure_columns:
        if target_failure in df.columns:
            print(f"\nAnalisando relação com '{target_failure}'...")
            plt.figure(figsize=(15, 10)) # Tamanho da figura ajustado para 6 boxplots

            for i, feature in enumerate(operational_features):
                if feature in df.columns:
                    plt.subplot(2, 3, i + 1) # Layout de 2 linhas e 3 colunas
                    sns.boxplot(x=target_failure, y=feature, data=df) # Usa o DataFrame original (ou processado com NaNs, que o boxplot trata)
                    plt.title(f'Distribuição de {feature.replace("_", " ").title()} por {target_failure.replace("(", "").replace(")", "").replace("Falha", "").strip()}')
                    plt.xlabel(target_failure.replace("(", "").replace(")", "").replace("Falha", "").strip())
                    plt.ylabel(feature.replace('_', ' ').title())
                else:
                     print(f"Aviso: Coluna operacional '{feature}' não encontrada para plotagem.")


            plt.tight_layout() # Ajusta o layout
            plt.show()
        else:
            print(f"Aviso: Coluna de falha '{target_failure}' não encontrada no DataFrame.")


    print("\nAnálise exploratória concluída.")


if __name__ == '__main__':
    caminho_dados = str(DATA_RAW_CSV)
    df_processado = preparar_dados(caminho_dados)

    # Realiza a análise exploratória com o DataFrame processado
    realizar_analise_exploratoria(df_processado)
