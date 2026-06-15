import sys
import json
import time
import pathlib

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

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

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)),
        ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced"))),
    ])

    start = time.time()
    pipeline.fit(train["text"], train["label"])
    train_time = time.time() - start

    y_pred = pipeline.predict(test["text"])

    path = MODELS / "svm.joblib"
    joblib.dump(pipeline, path)

    samples = test["text"].sample(min(100, len(test)), random_state=1).tolist()
    latency = measure_latency(lambda s: pipeline.predict([s]), samples)

    plot_confusion(test["label"], y_pred, CLASSES, ROOT / "svm_confusion.png")
    append_result({
        "model": "svm",
        "macro_f1": macro_f1(test["label"], y_pred),
        "f1_per_class": json.dumps(per_class_f1(test["label"], y_pred, CLASSES)),
        "train_time_s": round(train_time, 1),
        "latency_ms": latency,
        "model_size_mb": model_size_mb(path),
        "n_test": len(test),
    })

    print(report(test["label"], y_pred, CLASSES))
    print("svm macro-F1:", macro_f1(test["label"], y_pred), "| latency_ms:", latency)


if __name__ == "__main__":
    main()
