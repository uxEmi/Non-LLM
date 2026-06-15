"""Generate a synthetic English ticket dataset for local testing.

The real pipeline uses the CFPB Consumer Complaint Database (see shared/prepare_data.py).
That download isn't always available, so this script fabricates class-separable
complaint text — enough signal to train usable demo models, with deliberate
vocabulary overlap so the task isn't trivially perfect. Output contract matches
prepare_data.py exactly: data/train.csv / data/test.csv with columns text,label.

    python shared/make_demo_data.py --per-class 600
"""

import sys
import pathlib
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from shared.labels import CLASSES

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# Per-class building blocks. Each complaint = subject + problem + (optional) tail.
TEMPLATES = {
    "Loans": {
        "subject": ["my mortgage", "my auto loan", "my student loan", "my personal loan",
                    "my home loan", "the loan application", "my car finance", "my refinance"],
        "problem": ["was rejected without any explanation", "had a sudden interest rate increase",
                    "shows the wrong outstanding balance", "had an extra origination fee added",
                    "monthly installment was miscalculated", "payment was not applied to the principal",
                    "was denied even with a good credit score", "amortization schedule looks incorrect"],
        "tail": ["I want this loan reviewed.", "Please correct my repayment terms.",
                 "The lender never returned my call.", "I need the interest recalculated.", ""],
    },
    "Credit Reporting": {
        "subject": ["my credit report", "my credit score", "the credit bureau", "my consumer report",
                    "my credit file", "the reporting agency"],
        "problem": ["shows an account I never opened", "contains an inaccurate late payment",
                    "lists a fraudulent hard inquiry", "has a mixed file with someone else",
                    "dropped without any reason", "still shows a debt I already paid",
                    "was not updated after my dispute", "reports the wrong personal information"],
        "tail": ["Please remove this from my report.", "I dispute this entry.",
                 "This is hurting my creditworthiness.", "I want an investigation opened.", ""],
    },
    "Bank Accounts and Services": {
        "subject": ["my checking account", "my savings account", "my bank account", "a wire transfer",
                    "my direct deposit", "an ATM withdrawal", "my account statement", "a money transfer"],
        "problem": ["was charged unauthorized maintenance fees", "shows an overdraft fee I should not owe",
                    "was frozen without notice", "had a deposit that never posted",
                    "transfer was delayed for several days", "had a double withdrawal at the ATM",
                    "monthly service charge keeps appearing", "balance does not match my statement"],
        "tail": ["Please refund the fees.", "I need access to my funds.",
                 "The branch could not help me.", "Fix my account balance.", ""],
    },
    "Debt Collection": {
        "subject": ["a debt collector", "the collection agency", "a recovery firm", "the collector",
                    "a debt collection company", "an agent from collections"],
        "problem": ["calls me every day about a debt I do not recognize", "is harassing me at work",
                    "refuses to validate the debt", "is trying to collect a debt that is not mine",
                    "threatened me over an old balance", "keeps contacting me after I asked them to stop",
                    "reported the debt twice", "is collecting an amount I already settled"],
        "tail": ["Make them stop calling.", "I want debt validation.",
                 "This debt is not mine.", "Their behavior is illegal.", ""],
    },
    "Credit Card Services": {
        "subject": ["my credit card", "my card account", "the card issuer", "my rewards card",
                    "my credit card statement", "the annual fee on my card"],
        "problem": ["charged me twice for the annual fee", "has an unauthorized charge I did not make",
                    "rewards points were never credited", "interest was charged after I paid in full",
                    "billing statement shows a duplicate transaction", "raised my APR without notice",
                    "declined a refund for a returned item", "late fee was applied unfairly"],
        "tail": ["I want my money back.", "Please reverse this charge.",
                 "Refund the annual fee.", "Correct my billing statement.", ""],
    },
}

# A little shared filler so classes share some neutral vocabulary (realistic noise).
PREFIX = ["", "", "Hello, ", "To whom it may concern, ", "I am writing because ",
          "I am very frustrated that ", "For the third time, "]


def make_rows(label, n, rng):
    blocks = TEMPLATES[label]
    rows = []
    for _ in range(n):
        prefix = rng.choice(PREFIX)
        subject = rng.choice(blocks["subject"])
        problem = rng.choice(blocks["problem"])
        tail = rng.choice(blocks["tail"])
        sentence = f"{prefix}{subject} {problem}.".strip()
        if sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        text = (sentence + " " + tail).strip()
        rows.append({"text": text, "label": label})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=600)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for label in CLASSES:
        rows += make_rows(label, args.per_class, rng)

    frame = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train, test = train_test_split(
        frame, test_size=args.test_size, stratify=frame["label"], random_state=args.seed
    )
    train.to_csv(DATA / "train.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)

    print(f"train={len(train)}  test={len(test)}")
    print(train["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
