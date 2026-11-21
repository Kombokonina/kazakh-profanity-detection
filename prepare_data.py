"""Step 1: remove duplicates, group near-duplicates, split the data and apply the label fixes.

Run:  python prepare_data.py
Reads data/kazakh_profanity_share.csv and data/label_fixes.csv.
Writes data/processed/train.csv, val.csv, test.csv and split_summary.json.
"""
import json
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold

SEED = 5113
NEAR_DUP_JACCARD = 0.8
# KazSAnDRA id prefix -> review domain
SOURCES = {"pla": "appstore", "kas": "market", "fli": "market",
           "gis": "mapping", "map": "mapping", "kit": "bookstore"}


def duplicate_key(text):
    """Two comments are duplicates if only letter case or spaces differ."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def letters_only(text):
    """'Көтттт!!' -> 'көт': letters only, repeated letters collapsed."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\W\d_]+", " ", text)
    text = re.sub(r"(\w)\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def near_duplicate_groups(texts):
    """Same group id for comments whose 3-letter pieces overlap by at least 80% (Jaccard)."""
    pieces = CountVectorizer(analyzer="char", ngram_range=(3, 3), binary=True, lowercase=False)
    X = pieces.fit_transform([f" {letters_only(t)} " for t in texts])
    sizes = X.sum(axis=1).A1
    shared = (X @ X.T).tocoo()                 # number of shared pieces for every pair

    parent = list(range(len(texts)))           # join near-duplicate pairs into groups

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, n in zip(shared.row, shared.col, shared.data):
        if i < j and n / (sizes[i] + sizes[j] - n) >= NEAR_DUP_JACCARD:
            parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def counts(part):
    return {"rows": len(part), "label_0": int((part["label"] == 0).sum()), "label_1": int((part["label"] == 1).sum())}


def main():
    df = pd.read_csv("data/kazakh_profanity_share.csv").rename(columns={"split": "shared_split"})
    df["source"] = df["kazsandra_id"].str[:3].map(SOURCES)
    n_raw = len(df)

    # 1. Remove exact duplicates
    df["dup_key"] = df["text"].map(duplicate_key)
    removed = df[df.duplicated("dup_key")]
    df = df.drop_duplicates("dup_key").reset_index(drop=True)

    # 2. Near-duplicates get one group id, so they always land in the same split
    df["group"] = near_duplicate_groups(df["text"].tolist())
    sizes = df["group"].value_counts()

    # 3. Split by group, keeping the share of profane comments: 1 of 10 parts test, 1 val, 8 train
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    parts = [idx for _, idx in splitter.split(df, df["label"], df["group"])]
    df["split"] = "train"
    df.loc[parts[0], "split"] = "test"
    df.loc[parts[1], "split"] = "val"
    assert df.groupby("group")["split"].nunique().max() == 1      # no group crosses splits

    # 4. Label fixes (reasons in data/label_fixes.csv)
    fixes = pd.read_csv("data/label_fixes.csv")
    for _, fix in fixes.iterrows():
        row = df["kazsandra_id"] == fix["kazsandra_id"]
        assert df.loc[row, "label"].tolist() == [fix["old_label"]], fix["kazsandra_id"]
        df.loc[row, "label"] = fix["new_label"]

    columns = ["kazsandra_id", "text", "label", "source", "group", "split"]
    for name in ["train", "val", "test"]:
        df.loc[df["split"] == name, columns].to_csv(f"data/processed/{name}.csv", index=False)

    summary = {
        "seed": SEED,
        "rows_in_shared_file": n_raw,
        "exact_duplicates_removed": len(removed),
        "removed_ids": removed["kazsandra_id"].tolist(),
        "rows_after_deduplication": len(df),
        "near_duplicate_groups_with_2_or_more_rows": int((sizes > 1).sum()),
        "rows_in_those_groups": int(sizes[sizes > 1].sum()),
        "label_fixes_applied": len(fixes),
        "splits": {name: counts(df[df["split"] == name]) for name in ["train", "val", "test"]},
        "total": counts(df),
    }
    with open("data/processed/split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
