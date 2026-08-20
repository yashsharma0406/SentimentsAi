import pandas as pd
from src.config import DATA_PATH
def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    # Validate dataset path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Ensure the file is committed to GitHub and exists in /data."
        )
    print(f"Loading dataset from: {path}")
    # Load dataset
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise ValueError("Dataset loaded but is empty.")

    # Required columns check
    required_cols = {"review", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    # Clean + normalize
    df = df.drop_duplicates().copy()

    # Normalize labels safely
    label_map = {"pos": 1, "neg": 0, 1: 1, 0: 0}
    df["label"] = df["label"].map(label_map)

    if df["label"].isna().any():
        raise ValueError("Invalid labels detected. Allowed values: pos / neg / 1 / 0")
    df["review"] = df["review"].astype(str)
    df["label"] = df["label"].astype(int)

    print(f"Dataset loaded successfully: {len(df)} rows")

    return df
