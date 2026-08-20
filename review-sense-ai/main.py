import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (OUTPUTS_DIR, MODEL_PATH,TEST_SIZE,RANDOM_STATE,DATA_PATH,)
from src.data_load import load_dataset
from src.preprocess import preprocess_text
from src.modeling_compare import compare_tfidf, compare_bert
from src.calibrate_train import train_calibrated_svm
from src.error_analysis import misclassified_report
from src.utils import save_joblib, save_json, ensure_dir

# SAFETY CHECK: dataset must exist for training
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset missing at {DATA_PATH}. "
        "Make sure you committed it inside the repo under /data."
    )
def main():
    # Ensure folders exist
    ensure_dir(OUTPUTS_DIR)
    ensure_dir(MODEL_PATH.parent)

    # Load dataset
    df = load_dataset()

    # Basic cleanup
    df["review"] = df["review"].astype(str)
    df["label"] = df["label"].astype(int)

    # Clean text for modeling + insights
    df["clean_review"] = df["review"].apply(preprocess_text)

    X = df["review"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE,stratify=y)

    # Clean splits
    X_train_clean = X_train.apply(preprocess_text)
    X_test_clean = X_test.apply(preprocess_text)

    # 1) Compare models (writes outputs/reports/model_comparison.csv)
    tfidf_df = compare_tfidf(X_train_clean, X_test_clean, y_train, y_test)
    bert_df = compare_bert(X_train_clean, X_test_clean, y_train, y_test)

    results_df = (
        pd.concat([tfidf_df, bert_df], ignore_index=True)
        .sort_values(by=["F1", "Accuracy"], ascending=False)
        .reset_index(drop=True)
    )

    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    print(f"Saved model comparison to: {OUTPUTS_DIR / 'model_comparison.csv'}")
    print("\nTop models:\n", results_df.head(5))

    # 2) Train final calibrated model
    calibrated_model, y_pred, y_proba, metrics = train_calibrated_svm(X_train_clean, y_train, X_test_clean, y_test)
    save_json(metrics, OUTPUTS_DIR / "calibrated_metrics.json")
    save_joblib(calibrated_model, MODEL_PATH)
    print(f"Saved calibrated model to: {MODEL_PATH}")

    # 3) Misclassification insights
    df_all, mis, summary = misclassified_report(X_test,X_test_clean,y_test,y_pred,y_proba)
    mis.to_csv(OUTPUTS_DIR / "misclassified.csv", index=False)
    save_json(summary, OUTPUTS_DIR / "misclassified_summary.json")
    print("\nMisclassification summary:")
    print(summary)

if __name__ == "__main__":
    main()
