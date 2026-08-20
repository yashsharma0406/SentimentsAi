from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "amazonreviews.tsv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "reports"

MODEL_PATH = ARTIFACTS_DIR / "best_model_calibrated.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
