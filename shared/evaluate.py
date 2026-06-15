import os
import csv
import time
import datetime
import pathlib

from sklearn.metrics import f1_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results.csv"
FIELDS = ["model", "macro_f1", "f1_per_class", "train_time_s",
          "latency_ms", "model_size_mb", "n_test", "timestamp"]


def macro_f1(y_true, y_pred):
    return round(float(f1_score(y_true, y_pred, average="macro")), 4)


def per_class_f1(y_true, y_pred, labels):
    scores = f1_score(y_true, y_pred, average=None, labels=labels)
    return {label: round(float(s), 3) for label, s in zip(labels, scores)}


def report(y_true, y_pred, labels):
    return classification_report(y_true, y_pred, labels=labels, digits=3)


def measure_latency(predict_fn, samples, repeats=3):
    timings = []
    for _ in range(repeats):
        for sample in samples:
            start = time.perf_counter()
            predict_fn(sample)
            timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return round(timings[len(timings) // 2], 2)


def model_size_mb(path):
    return round(os.path.getsize(path) / 1_000_000, 2)


def plot_confusion(y_true, y_pred, labels, out_path):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="mako",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.ylabel("Adevăr")
    plt.xlabel("Predicție")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def append_result(row):
    row.setdefault("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    is_new = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
