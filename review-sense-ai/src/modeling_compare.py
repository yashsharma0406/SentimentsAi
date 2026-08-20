import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

def compare_tfidf(X_train, X_test, y_train, y_test):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=3000), {"clf__C": [0.1, 1, 10]}),
        "LinearSVM": (LinearSVC(), {"clf__C": [0.1, 1, 10]}),
        "NeuralNetwork": (MLPClassifier(max_iter=300), {"clf__hidden_layer_sizes": [(128,), (256,)],
                                                        "clf__alpha": [0.0001, 0.001]})
    }

    rows = []
    for name, (model, params) in models.items():
        pipeline = Pipeline([("tfidf", tfidf), ("clf", model)])
        grid = GridSearchCV(pipeline, param_grid=params, scoring="f1", cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        rows.append({
            "Features": "TF-IDF",
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "Best Params": grid.best_params_
        })

    return pd.DataFrame(rows)

def compare_bert(X_train_clean, X_test_clean, y_train, y_test):
    from sentence_transformers import SentenceTransformer

    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train_emb = bert_model.encode(X_train_clean.tolist(), batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    X_test_emb = bert_model.encode(X_test_clean.tolist(), batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=5000, class_weight="balanced"), {"clf__C": [0.01, 0.1, 1, 10]}),
        "SVM_RBF": (SVC(kernel="rbf", probability=True, class_weight="balanced"),
                    {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}),
        "NeuralNetwork": (MLPClassifier(max_iter=400, early_stopping=True, random_state=42),
                          {"clf__hidden_layer_sizes": [(128,), (256,), (256, 128)], "clf__alpha": [0.0001, 0.001]})
    }

    rows = []
    for name, (model, params) in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        grid = GridSearchCV(pipeline, param_grid=params, scoring="f1", cv=3, n_jobs=-1)
        grid.fit(X_train_emb, y_train)

        y_pred = grid.predict(X_test_emb)
        rows.append({
            "Features": "BERT",
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "Best Params": grid.best_params_
        })

    return pd.DataFrame(rows)
