from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

def train_calibrated_svm(X_train_clean, y_train, X_test_clean, y_test):
    base_pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                              ("svm", LinearSVC(C=0.1))])

    calibrated = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=3)
    calibrated.fit(X_train_clean, y_train)

    y_pred = calibrated.predict(X_test_clean)
    y_proba = calibrated.predict_proba(X_test_clean)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics = {
        "accuracy": float(round(acc, 4)),
        "f1": float(round(f1, 4)),
        "roc_auc": float(round(auc, 4)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    }
    return calibrated, y_pred, y_proba, metrics
