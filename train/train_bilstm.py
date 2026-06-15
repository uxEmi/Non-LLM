import sys
import json
import time
import pathlib
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from shared.labels import CLASSES
from shared.evaluate import (macro_f1, per_class_f1, model_size_mb,
                             plot_confusion, append_result, report)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 200
EPOCHS = 5


def tokenize(text):
    return str(text).lower().split()


def build_vocab(texts, max_size=20000, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    itos = ["<pad>", "<unk>"] + [w for w, f in counter.most_common(max_size) if f >= min_freq]
    return {word: index for index, word in enumerate(itos)}


def encode(text, stoi):
    ids = [stoi.get(word, 1) for word in tokenize(text)][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


class TicketDataset(Dataset):
    def __init__(self, texts, labels, stoi):
        self.encoded = [encode(t, stoi) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.encoded[index]), torch.tensor(self.labels[index])


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


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    class_to_index = {label: index for index, label in enumerate(CLASSES)}
    index_to_class = {index: label for label, index in class_to_index.items()}

    x_train = train["text"].tolist()
    y_train = [class_to_index[label] for label in train["label"]]
    x_test = test["text"].tolist()
    y_test = [class_to_index[label] for label in test["label"]]

    stoi = build_vocab(x_train)
    train_loader = DataLoader(TicketDataset(x_train, y_train, stoi), batch_size=64, shuffle=True)
    test_loader = DataLoader(TicketDataset(x_test, y_test, stoi), batch_size=128)

    model = BiLSTMClassifier(len(stoi), n_classes=len(CLASSES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch + 1}/{EPOCHS}  loss={running / len(train_loader):.4f}")
    train_time = time.time() - start

    model.eval()
    predictions = []
    latencies = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(DEVICE)
            tick = time.perf_counter()
            logits = model(batch_x)
            latencies.append((time.perf_counter() - tick) / len(batch_x) * 1000)
            predictions += logits.argmax(dim=1).cpu().tolist()

    y_pred = [index_to_class[p] for p in predictions]
    y_true = [index_to_class[i] for i in y_test]

    path = MODELS / "bilstm.pt"
    torch.save({"state": model.state_dict(), "stoi": stoi, "classes": CLASSES}, path)

    latency = round(sorted(latencies)[len(latencies) // 2], 2)
    plot_confusion(y_true, y_pred, CLASSES, ROOT / "bilstm_confusion.png")
    append_result({
        "model": "bilstm",
        "macro_f1": macro_f1(y_true, y_pred),
        "f1_per_class": json.dumps(per_class_f1(y_true, y_pred, CLASSES)),
        "train_time_s": round(train_time, 1),
        "latency_ms": latency,
        "model_size_mb": model_size_mb(path),
        "n_test": len(y_true),
    })

    print(report(y_true, y_pred, CLASSES))
    print("bilstm macro-F1:", macro_f1(y_true, y_pred), "| latency_ms:", latency)


if __name__ == "__main__":
    main()
