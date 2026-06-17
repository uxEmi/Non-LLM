import sys
import time
import math
import random
import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from shared.labels import CLASSES

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

app = FastAPI(title="Rutare tichete")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOADED = {}

HANDLE_MIN = {
    "Credit Reporting": 4,
    "Loans": 5,
    "Debt Collection": 3,
    "Credit Card Services": 3,
    "Bank Accounts and Services": 2,
}
QUEUE = {
    "Credit Reporting": 3.0,
    "Loans": 2.0,
    "Debt Collection": 2.0,
    "Credit Card Services": 1.0,
    "Bank Accounts and Services": 1.0,
}
DRAIN_PER_MIN = 0.8
LAST_DRAIN = {"t": time.time()}


def department_wait(team):
    now = time.time()
    elapsed_min = (now - LAST_DRAIN["t"]) / 60.0
    LAST_DRAIN["t"] = now
    for name in QUEUE:
        QUEUE[name] = max(0.0, QUEUE[name] - elapsed_min * DRAIN_PER_MIN)
    QUEUE[team] = QUEUE.get(team, 0.0) + 1.0
    pending = QUEUE[team]
    eta = max(1, math.ceil(pending * HANDLE_MIN.get(team, 3)))
    return eta, int(round(pending))


class Ticket(BaseModel):
    text: str
    model: str = "svm"


MAX_LEN = 200


def tokenize(text):
    return str(text).lower().split()


def encode(text, stoi):
    ids = [stoi.get(word, 1) for word in tokenize(text)][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden=128, n_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        pooled = output.max(dim=1).values
        return self.fc(self.dropout(pooled))


def load_models():
    svm_path = MODELS / "svm.joblib"
    xgb_path = MODELS / "xgboost.joblib"
    bilstm_path = MODELS / "bilstm.pt"
    if svm_path.exists():
        LOADED["svm"] = joblib.load(svm_path)
    if xgb_path.exists():
        LOADED["xgboost"] = joblib.load(xgb_path)
    if bilstm_path.exists():
        checkpoint = torch.load(bilstm_path, map_location="cpu")
        stoi = checkpoint["stoi"]
        classes = checkpoint["classes"]
        model = BiLSTMClassifier(len(stoi), n_classes=len(classes))
        model.load_state_dict(checkpoint["state"])
        model.eval()
        LOADED["bilstm"] = {"model": model, "stoi": stoi, "classes": classes}


def translate_to_english(text):
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def proba_svm(text):
    pipeline = LOADED["svm"]
    probabilities = pipeline.predict_proba([text])[0]
    return {cls: float(p) for cls, p in zip(pipeline.classes_, probabilities)}


def proba_xgboost(text):
    bundle = LOADED["xgboost"]
    features = bundle["vectorizer"].transform([text])
    probabilities = bundle["classifier"].predict_proba(features)[0]
    return {cls: float(p) for cls, p in zip(bundle["encoder"].classes_, probabilities)}


def proba_bilstm(text):
    bundle = LOADED["bilstm"]
    ids = torch.tensor([encode(text, bundle["stoi"])])
    with torch.no_grad():
        probabilities = torch.softmax(bundle["model"](ids), dim=1)[0].tolist()
    return {cls: float(p) for cls, p in zip(bundle["classes"], probabilities)}


def model_proba(model, text):
    if model == "xgboost" and "xgboost" in LOADED:
        return proba_xgboost(text)
    if model == "svm" and "svm" in LOADED:
        return proba_svm(text)
    if model == "bilstm" and "bilstm" in LOADED:
        return proba_bilstm(text)
    return None


def _top_scored(scored, k):
    scored = [s for s in scored if s[1] > 0]
    scored.sort(key=lambda s: s[1], reverse=True)
    out, seen = [], set()
    for word, _ in scored:
        if word not in seen:
            seen.add(word)
            out.append(word)
        if len(out) >= k:
            break
    return out


def _scored_features(model, text, team):
    if model == "svm" and "svm" in LOADED:
        pipeline = LOADED["svm"]
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["svm"]
        classes = list(pipeline.classes_)
        if team not in classes:
            return []
        ci = classes.index(team)
        coef = np.mean([cc.estimator.coef_[ci] for cc in clf.calibrated_classifiers_], axis=0)
        x = tfidf.transform([text])
        names = tfidf.get_feature_names_out()
        return [(names[idx], float(val * coef[idx])) for idx, val in zip(x.indices, x.data)]
    if model == "xgboost" and "xgboost" in LOADED:
        import xgboost as xgb
        bundle = LOADED["xgboost"]
        vectorizer, classifier, encoder = bundle["vectorizer"], bundle["classifier"], bundle["encoder"]
        classes = list(encoder.classes_)
        if team not in classes:
            return []
        ci = classes.index(team)
        x = vectorizer.transform([text])
        contribs = np.array(classifier.get_booster().predict(xgb.DMatrix(x), pred_contribs=True))
        row = contribs[0, ci, :-1]
        names = vectorizer.get_feature_names_out()
        return [(names[idx], float(row[idx])) for idx in x.indices]
    return []


def explain(model, text, team):
    try:
        scored = _scored_features(model, text, team)
    except Exception:
        return [], {}
    top = _top_scored(scored, 5)
    unigrams = [(name, w) for name, w in scored if " " not in name]
    max_abs = max((abs(w) for _, w in unigrams), default=0.0)
    weights = {}
    if max_abs > 0:
        for name, w in unigrams:
            weights[name] = round(w / max_abs, 3)
    return top, weights


load_models()


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(LOADED.keys())}


@app.post("/predict")
def predict(ticket: Ticket):
    text_en = translate_to_english(ticket.text)
    start = time.perf_counter()
    probs = model_proba(ticket.model, text_en)
    if probs is None:
        team = random.choice(CLASSES)
        confidence = 0.5
        probs = {c: (0.5 if c == team else 0.125) for c in CLASSES}
    else:
        team = max(probs, key=probs.get)
        confidence = probs[team]

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    wait_min, queue_len = department_wait(team)
    top_words, word_weights = explain(ticket.model, text_en, team)
    return {
        "predicted_team": team,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "model": ticket.model,
        "estimated_wait_min": wait_min,
        "queue_len": queue_len,
        "top_words": top_words,
        "word_weights": word_weights,
        "text_en": text_en,
        "probabilities": probs,
    }


@app.post("/trace")
def trace(ticket: Ticket):
    text_en = translate_to_english(ticket.text)
    tokens = text_en.split()
    steps = [{"n": 0, "word": "", "probabilities": {c: 1.0 / len(CLASSES) for c in CLASSES}}]
    if tokens:
        n = len(tokens)
        max_steps = 40
        if n <= max_steps:
            lengths = list(range(1, n + 1))
        else:
            lengths = sorted({max(1, round((i + 1) * n / max_steps)) for i in range(max_steps)})
            lengths[-1] = n
        for k in lengths:
            probs = model_proba(ticket.model, " ".join(tokens[:k]))
            if probs is None:
                steps = []
                break
            steps.append({"n": k, "word": tokens[k - 1], "probabilities": probs})
    return {"text_en": text_en, "model": ticket.model, "steps": steps}


@app.post("/predict_all")
def predict_all(ticket: Ticket):
    text_en = translate_to_english(ticket.text)
    models = {}
    for name in ("svm", "xgboost", "bilstm"):
        if name not in LOADED:
            continue
        start = time.perf_counter()
        probs = model_proba(name, text_en)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        team = max(probs, key=probs.get)
        models[name] = {
            "predicted_team": team,
            "confidence": probs[team],
            "latency_ms": latency_ms,
            "probabilities": probs,
        }
    return {"text_en": text_en, "models": models}
