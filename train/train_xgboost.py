import sys
import json
import time
import pathlib

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from shared.labels import CLASSES
from shared.evaluate import (macro_f1, per_class_f1, measure_latency,
                             model_size_mb, plot_confusion, append_result, report)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                 sublinear_tf=True, max_features=40000)
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])

    encoder = LabelEncoder().fit(CLASSES)
    y_train = encoder.transform(train["label"])

    classifier = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.2,
        subsample=0.9, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
    )

    start = time.time()
    classifier.fit(x_train, y_train)
    train_time = time.time() - start

    y_pred = encoder.inverse_transform(classifier.predict(x_test))

    path = MODELS / "xgboost.joblib"
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "encoder": encoder}, path)

    def predict_one(text):
        return encoder.inverse_transform(classifier.predict(vectorizer.transform([text])))[0]

    samples = test["text"].sample(min(100, len(test)), random_state=1).tolist()
    latency = measure_latency(predict_one, samples)

    plot_confusion(test["label"], y_pred, CLASSES, ROOT / "xgboost_confusion.png")
    append_result({
        "model": "xgboost",
        "macro_f1": macro_f1(test["label"], y_pred),
        "f1_per_class": json.dumps(per_class_f1(test["label"], y_pred, CLASSES)),
        "train_time_s": round(train_time, 1),
        "latency_ms": latency,
        "model_size_mb": model_size_mb(path),
        "n_test": len(test),
    })

    print(report(test["label"], y_pred, CLASSES))
    print("xgboost macro-F1:", macro_f1(test["label"], y_pred), "| latency_ms:", latency)


if __name__ == "__main__":
    main()
