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


def predict_svm(text):
    pipeline = LOADED["svm"]
    probabilities = pipeline.predict_proba([text])[0]
    index = probabilities.argmax()
    return pipeline.classes_[index], float(probabilities[index])


def predict_xgboost(text):
    bundle = LOADED["xgboost"]
    features = bundle["vectorizer"].transform([text])
    probabilities = bundle["classifier"].predict_proba(features)[0]
    index = probabilities.argmax()
    return bundle["encoder"].inverse_transform([index])[0], float(probabilities[index])


def predict_bilstm(text):
    bundle = LOADED["bilstm"]
    ids = torch.tensor([encode(text, bundle["stoi"])])
    with torch.no_grad():
        probabilities = torch.softmax(bundle["model"](ids), dim=1)[0]
    index = int(probabilities.argmax())
    return bundle["classes"][index], float(probabilities[index])


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


def words_svm(text, team, k=5):
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
    scored = [(names[idx], float(val * coef[idx])) for idx, val in zip(x.indices, x.data)]
    return _top_scored(scored, k)


def words_xgboost(text, team, k=5):
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
    scored = [(names[idx], float(row[idx])) for idx in x.indices]
    return _top_scored(scored, k)


def influential_words(model, text, team, k=5):
    try:
        if model == "svm" and "svm" in LOADED:
            return words_svm(text, team, k)
        if model == "xgboost" and "xgboost" in LOADED:
            return words_xgboost(text, team, k)
    except Exception:
        return []
    return []


load_models()


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(LOADED.keys())}


@app.post("/predict")
def predict(ticket: Ticket):
    text_en = translate_to_english(ticket.text)
    start = time.perf_counter()

    if ticket.model == "xgboost" and "xgboost" in LOADED:
        team, confidence = predict_xgboost(text_en)
    elif ticket.model == "svm" and "svm" in LOADED:
        team, confidence = predict_svm(text_en)
    elif ticket.model == "bilstm" and "bilstm" in LOADED:
        team, confidence = predict_bilstm(text_en)
    else:
        team, confidence = random.choice(CLASSES), 0.5

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    wait_min, queue_len = department_wait(team)
    return {
        "predicted_team": team,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "model": ticket.model,
        "estimated_wait_min": wait_min,
        "queue_len": queue_len,
        "top_words": influential_words(ticket.model, text_en, team),
    }
