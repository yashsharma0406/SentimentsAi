from collections import Counter
from src.utils import save_json
from src.config import OUTPUTS_DIR
def top_words(df, col="clean_review", label_col="label", n=20):
    pos = Counter(" ".join(df[df[label_col] == 1][col]).split()).most_common(n)
    neg = Counter(" ".join(df[df[label_col] == 0][col]).split()).most_common(n)
    return {"positive": pos, "negative": neg}
def run_eda(df):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    words = top_words(df)
    save_json(words, OUTPUTS_DIR / "top_words.json")
    return words
