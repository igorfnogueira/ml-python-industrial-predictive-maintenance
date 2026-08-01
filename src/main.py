from data_preparation import preparar_dados
from model_training import treinar_modelos
from prediction_generation import gerar_predicoes_csv
from config import DATA_RAW_CSV, DEFAULT_PREDICTIONS_CSV, MODELS_DIR, OUTPUTS_DIR


def main():
    """
    Orquestra o fluxo completo do projeto de manutenção preditiva:
    preparação de dados, treinamento de modelos e geração de CSV com predições.
    """
    print("Iniciando o fluxo principal do projeto de manutenção preditiva...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    caminho_arquivo_dados = str(DATA_RAW_CSV)
    caminho_saida_predicoes = str(DEFAULT_PREDICTIONS_CSV)

    print("\nExecutando etapa de Preparação e Limpeza dos Dados...")
    df_processado = preparar_dados(caminho_arquivo_dados)
    print("Etapa de Preparação e Limpeza dos Dados concluída.")

    print("\nExecutando etapa de Treinamento de Modelos...")
    pipelines_treinados, relatorios_treinamento, predicoes_teste = treinar_modelos(df_processado)
    print("Etapa de Treinamento de Modelos concluída.")

    print("\nExecutando etapa de Geração de Arquivo CSV com Predições...")
    gerar_predicoes_csv(
        df_processado,
        pipelines_treinados,
        caminho_saida_csv=caminho_saida_predicoes,
    )
    print(f"Etapa de Geração de Arquivo CSV com Predições concluída. Arquivo salvo em: {caminho_saida_predicoes}")

    print("\nFluxo principal do projeto concluído.")


if __name__ == "__main__":
    main()
