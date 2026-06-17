import sys
import pathlib
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from shared.labels import CLASSES

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

TEXT_COLS = ["narrative", "Consumer complaint narrative", "consumer_complaint_narrative"]
LABEL_COLS = ["product_5", "Product", "product"]


def consolidate(product):
    p = str(product).lower()
    if "debt collection" in p:
        return "Debt Collection"
    if "credit report" in p or "personal consumer report" in p:
        return "Credit Reporting"
    if "credit card" in p or "prepaid" in p:
        return "Credit Card Services"
    if any(k in p for k in ["mortgage", "loan", "lease", "student"]):
        return "Loans"
    if any(k in p for k in ["bank account", "checking", "savings",
                            "money transfer", "money service", "virtual currency"]):
        return "Bank Accounts and Services"
    return None


def pick_column(candidates, frame):
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=60000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    text_col = pick_column(TEXT_COLS, frame)
    label_col = pick_column(LABEL_COLS, frame)
    if text_col is None or label_col is None:
        raise SystemExit("Coloane lipsă. Coloane găsite: " + ", ".join(map(str, frame.columns[:25])))

    frame = frame[[text_col, label_col]].dropna()
    frame.columns = ["text", "label_raw"]

    if label_col == "product_5" and set(frame["label_raw"].unique()).issubset(set(CLASSES)):
        frame["label"] = frame["label_raw"]
    else:
        frame["label"] = frame["label_raw"].map(consolidate)

    frame = frame.dropna(subset=["label"])
    frame = frame[frame["text"].str.len() > 20]

    per_class = max(1, args.n // len(CLASSES))
    if len(frame) > args.n:
        groups = [
            group.sample(min(len(group), per_class), random_state=args.seed)
            for _, group in frame.groupby("label")
        ]
        frame = pd.concat(groups).sample(frac=1, random_state=args.seed)


    train, test = train_test_split(
        frame[["text", "label"]],
        test_size=args.test_size,
        stratify=frame["label"],
        random_state=args.seed,
    )

    train.to_csv(DATA / "train.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)

    print(f"train={len(train)}  test={len(test)}")
    print(train["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
