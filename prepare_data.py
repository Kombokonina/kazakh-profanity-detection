"""Step 1: clean the shared data, fix labels, remove duplicates, group near-duplicates and split.

Run:  python prepare_data.py
Reads data/kazakh_profanity_share.csv and data/label_fixes.csv,
writes data/processed/{train,val,test}.csv and data/processed/split_summary.json.
"""
import json
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold

SEED = 5113
RAW_FILE = "data/kazakh_profanity_share.csv"
FIXES_FILE = "data/label_fixes.csv"
OUT_DIR = "data/processed"
NEAR_DUP_JACCARD = 0.8
# KazSAnDRA id prefix -> review domain (checked against KazSAnDRA's own "domain" column)
SOURCES = {"pla": "appstore", "kas": "market", "fli": "market",
           "gis": "mapping", "map": "mapping", "kit": "bookstore"}


def duplicate_key(text):
    """Two comments are duplicates if only letter case or spaces differ."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def letters_only(text):
    """Keep letters only and collapse repeated letters: 'Көтттт!!' -> 'көт'."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\W\d_]+", " ", text)
    text = re.sub(r"(\w)\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def near_duplicate_groups(texts):
    """Give near-duplicates the same group id.

    Two comments are near-duplicates if the sets of 3-letter pieces of their
    letters_only() forms overlap by at least NEAR_DUP_JACCARD (Jaccard index).
    """
    padded = [f" {letters_only(t)} " for t in texts]
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), binary=True, lowercase=False)
    X = vectorizer.fit_transform(padded)
    sizes = X.sum(axis=1).A1
    shared = (X @ X.T).tocoo()                  # number of shared trigrams for every pair

    parent = list(range(len(texts)))            # union-find: join pairs into groups

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, n in zip(shared.row, shared.col, shared.data):
        if i < j and n / (sizes[i] + sizes[j] - n) >= NEAR_DUP_JACCARD:
            parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def main():
    df = pd.read_csv(RAW_FILE).rename(columns={"split": "shared_split"})
    df["source"] = df["kazsandra_id"].str[:3].map(SOURCES)
    n_raw = len(df)

    # 1. Remove exact duplicates (keep the first copy)
    df["dup_key"] = df["text"].map(duplicate_key)
    removed = df[df.duplicated("dup_key")]
    df = df.drop_duplicates("dup_key").reset_index(drop=True)

    # 2. Near-duplicates share a group id, so they always end up in the same split
    df["group"] = near_duplicate_groups(df["text"].tolist())
    group_sizes = df["group"].value_counts()

    # 3. Stratified split by group: 10 folds -> 1 for test, 1 for validation, 8 for training
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = [idx for _, idx in splitter.split(df, df["label"], df["group"])]
    df["split"] = "train"
    df.loc[folds[0], "split"] = "test"
    df.loc[folds[1], "split"] = "val"
    assert df.groupby("group")["split"].nunique().max() == 1      # no group crosses splits

    # 4. Label corrections, decided by reading the comments (see data/label_fixes.csv)
    fixes = pd.read_csv(FIXES_FILE)
    for _, fix in fixes.iterrows():
        rows = df.index[df["kazsandra_id"] == fix["kazsandra_id"]]
        assert len(rows) == 1 and df.at[rows[0], "label"] == fix["old_label"], fix["kazsandra_id"]
        df.at[rows[0], "label"] = fix["new_label"]

    columns = ["kazsandra_id", "text", "label", "source", "group", "split"]
    for name in ["train", "val", "test"]:
        df.loc[df["split"] == name, columns].to_csv(f"{OUT_DIR}/{name}.csv", index=False)

    summary = {
        "seed": SEED,
        "rows_in_shared_file": n_raw,
        "exact_duplicates_removed": len(removed),
        "removed_ids": removed["kazsandra_id"].tolist(),
        "rows_after_deduplication": len(df),
        "near_duplicate_groups_with_2_or_more_rows": int((group_sizes > 1).sum()),
        "rows_in_those_groups": int(group_sizes[group_sizes > 1].sum()),
        "label_fixes_applied": len(fixes),
        "splits": {name: {"rows": int((df["split"] == name).sum()),
                          "label_0": int(((df["split"] == name) & (df["label"] == 0)).sum()),
                          "label_1": int(((df["split"] == name) & (df["label"] == 1)).sum())}
                   for name in ["train", "val", "test"]},
        "total": {"rows": len(df), "label_0": int((df["label"] == 0).sum()), "label_1": int((df["label"] == 1).sum())},
    }
    with open(f"{OUT_DIR}/split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "removed_ids"}, indent=2))


if __name__ == "__main__":
    main()
