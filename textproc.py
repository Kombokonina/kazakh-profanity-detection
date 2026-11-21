"""Turn a comment into a list of character ids."""
import json
import re
import unicodedata
from collections import Counter

PAD, UNK = 0, 1     # id 0 = padding, id 1 = unknown character


def normalize(text):
    """'БЛЯЯЯЯЯ!!!!' -> 'бляя!!'. Punctuation and emoji stay: * and @ often hide swear words."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)        # 3 or more same characters -> 2
    return re.sub(r"\s+", " ", text).strip()


def build_vocab(texts, min_count=2):
    """Character -> id, for every character seen at least min_count times."""
    counts = Counter(ch for t in texts for ch in normalize(t))
    chars = sorted(ch for ch, n in counts.items() if n >= min_count)
    return {ch: i + 2 for i, ch in enumerate(chars)}   # ids 0 and 1 are taken


def encode(text, vocab, max_chars=None):
    ids = [vocab.get(ch, UNK) for ch in normalize(text)][:max_chars]
    return ids or [UNK]                                 # an empty comment becomes [UNK]


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pad_id": PAD, "unk_id": UNK, "chars": list(vocab)}, f, ensure_ascii=False, indent=0)


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        chars = json.load(f)["chars"]
    return {ch: i + 2 for i, ch in enumerate(chars)}
