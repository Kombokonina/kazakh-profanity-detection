"""Optional: check that every comment really comes from KazSAnDRA (so no example is synthetic).

Run:  git clone --depth 1 https://github.com/IS2AI/KazSAnDRA.git
      python verify_source.py KazSAnDRA
"""
import re
import sys
import unicodedata
import zipfile

import pandas as pd


def letters(text):
    """Compare texts by their letters only (case, punctuation, digits and repeats ignored)."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]|_|\d", " ", text)
    text = re.sub(r"(\w)\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def main(kazsandra_dir):
    parts = []
    for name in ["06_sc_train_ib", "09_sc_valid", "10_sc_test"]:      # together: all scored reviews
        with zipfile.ZipFile(f"{kazsandra_dir}/dataset/{name}.zip") as z:
            parts.append(pd.read_csv(z.open(f"{name}.csv")))
    source = pd.concat(parts).drop_duplicates("custom_id").set_index("custom_id")

    ours = pd.read_csv("data/kazakh_profanity_share.csv")
    masked = set(pd.read_csv("data/privacy_masks.csv")["kazsandra_id"])
    found = ours[ours["kazsandra_id"].isin(source.index)]
    differs = [i for i, t in zip(found["kazsandra_id"], found["text"])
               if letters(t) != letters(source.at[i, "text"])]
    explained = [i for i in differs
                 if i in masked or re.search(r"@user|<phone>", ours.loc[ours.kazsandra_id == i, "text"].iloc[0])]

    print(f"comments in our file:             {len(ours)}")
    print(f"ids found in KazSAnDRA:           {len(found)}")
    print(f"texts identical to the source:    {len(found) - len(differs)}")
    print(f"texts differing only by masking:  {len(explained)} (usernames, phone numbers or personal details)")
    print(f"other differences:                {[i for i in differs if i not in explained]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "KazSAnDRA")
