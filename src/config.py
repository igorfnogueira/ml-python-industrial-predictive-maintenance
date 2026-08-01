"""Caminhos padrão do projeto (relativos à raiz do repositório)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RAW_CSV = DATA_RAW_DIR / "bootcamp_train.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_PREDICTIONS_CSV = OUTPUTS_DIR / "predictions_classes.csv"


def model_path(failure_name: str) -> Path:
    """Monta o caminho do .pkl a partir do nome da coluna de falha."""
    safe = failure_name.replace(" ", "_").replace("(", "").replace(")", "")
    return MODELS_DIR / f"pipeline_{safe}.pkl"
