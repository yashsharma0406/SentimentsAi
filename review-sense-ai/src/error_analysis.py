import pandas as pd
def misclassified_report(X_test_raw, X_test_clean, y_test, y_pred, y_proba):
    df_err = pd.DataFrame({
        "review_raw": X_test_raw.values,
        "review_clean": X_test_clean.values,
        "true_label": y_test.values,
        "pred_label": y_pred,
        "proba_pos": y_proba
    })
    df_err["is_misclassified"] = df_err["true_label"] != df_err["pred_label"]
    df_err["confidence_margin"] = (df_err["proba_pos"] - 0.5).abs()
    df_err["length_chars"] = df_err["review_raw"].astype(str).str.len()
    df_err["length_tokens"] = df_err["review_clean"].astype(str).str.split().apply(len)

    mis = df_err[df_err["is_misclassified"]].copy()

    # False positives and false negatives
    fp = mis[(mis.true_label == 0) & (mis.pred_label == 1)].sort_values("confidence_margin", ascending=False)
    fn = mis[(mis.true_label == 1) & (mis.pred_label == 0)].sort_values("confidence_margin", ascending=False)

    summary = {
        "total_test": int(len(df_err)),
        "total_misclassified": int(len(mis)),
        "false_positives": int(len(fp)),
        "false_negatives": int(len(fn)),
        "top_fp_examples": fp.head(10)[["review_raw", "proba_pos"]].to_dict(orient="records"),
        "top_fn_examples": fn.head(10)[["review_raw", "proba_pos"]].to_dict(orient="records"),
    }

    return df_err, mis, summary
