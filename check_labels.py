"""Optional: the label checks described in the README, and a simple baseline for comparison.

Run:  python check_labels.py      (after prepare_data.py)
1. Keyword check, all rows: comments labelled 0 that contain a clearly obscene stem.
2. Baseline review, train + val only: a character TF-IDF + logistic regression model is trained
   with 5-fold cross-validation, and we list the 60 comments where it disagrees most with the label.
   The test set is not used here.
3. The same baseline trained on train and scored on validation, to compare with the CNN.
We read every listed comment by hand. The clear errors are in data/label_fixes.csv.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from textproc import normalize

SEED = 5113
OBSCENE = ["қотақ", "котак", "qotaq", "kotak", "сігейін", "сигейин", "шешеңнің амы", "амшық",
           "нахуй", "бля", "пизд", "ебан", "зайбал", "заибал", "пидар", "гандон", "сука"]


def baseline():
    return make_pipeline(TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(1, 5),
                                         sublinear_tf=True),
                         LogisticRegression(C=4, max_iter=2000))


def main():
    original = pd.read_csv("data/kazakh_profanity_share.csv").set_index("kazsandra_id")["label"]
    fixed = set(pd.read_csv("data/label_fixes.csv")["kazsandra_id"])
    parts = {s: pd.read_csv(f"data/processed/{s}.csv") for s in ["train", "val", "test"]}
    data = pd.concat(parts.values(), ignore_index=True)
    data["label"] = data["kazsandra_id"].map(original)            # labels before our fixes

    print("1. Label 0, but contains an obscene stem (all splits):")
    text = data["text"].map(normalize)
    hits = data[(data["label"] == 0) & text.apply(lambda t: any(w in t for w in OBSCENE))]
    for _, r in hits.iterrows():
        print(f"   {r['kazsandra_id']} {r['split']:5} {'FIXED' if r['kazsandra_id'] in fixed else '     '} {r['text'][:90]}")

    print("\n2. Baseline disagrees most with the label (train + val, 5-fold cross-validation):")
    tv = data[data["split"] != "test"].reset_index(drop=True)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(baseline(), tv["text"], tv["label"], cv=folds, method="predict_proba")[:, 1]
    tv["p"] = p
    tv["gap"] = np.abs(p - tv["label"])
    for _, r in tv.nlargest(60, "gap").iterrows():
        print(f"   label {r['label']}  baseline P={r['p']:.2f} {'FIXED' if r['kazsandra_id'] in fixed else '     '} "
              f"{r['text'][:80]}")

    print("\n3. Baseline trained on train, scored on validation (fixed labels, threshold 0.5):")
    model = baseline().fit(parts["train"]["text"], parts["train"]["label"])
    pred = (model.predict_proba(parts["val"]["text"])[:, 1] >= 0.5).astype(int)
    print(f"   validation F1 (profanity) = {f1_score(parts['val']['label'], pred):.3f}")


if __name__ == "__main__":
    main()
