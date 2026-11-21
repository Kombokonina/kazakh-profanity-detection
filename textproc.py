"""Text preprocessing: normalize a comment and turn it into character ids."""
import json
import re
import unicodedata
from collections import Counter

PAD, UNK = 0, 1          # id 0 = padding, id 1 = unknown character


def normalize(text):
    """Unicode-normalize, lowercase, shorten long repeats and tidy spaces.

    'БЛЯЯЯЯЯ!!!!'  ->  'бляя!!'
    Punctuation, digits and emoji are kept: '*' and '@' often hide swear words (с*ка, қот@қ).
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)        # 3 or more same characters -> 2
    return re.sub(r"\s+", " ", text).strip()


class CharVocab:
    """Maps each character to an id. Built from the training set only."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.index = {ch: i + 2 for i, ch in enumerate(self.chars)}   # 0 and 1 are reserved

    @classmethod
    def build(cls, texts, min_count=2):
        counts = Counter(ch for t in texts for ch in normalize(t))
        return cls(sorted(ch for ch, n in counts.items() if n >= min_count))

    def __len__(self):
        return len(self.chars) + 2

    def encode(self, text, max_chars=None):
        ids = [self.index.get(ch, UNK) for ch in normalize(text)]
        ids = ids[:max_chars] if max_chars else ids
        return ids or [UNK]                              # an empty comment becomes one unknown id

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"pad_id": PAD, "unk_id": UNK, "chars": self.chars}, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])
