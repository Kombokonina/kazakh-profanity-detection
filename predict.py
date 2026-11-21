"""Step 4: load the saved model and predict on new comments (no retraining needed).

Run:  python predict.py "Керемет ойын, маған ұнады" "Мал ойын"     (one or more comments)
      python predict.py                                            (type comments one by one)
"""
import json
import sys

import torch

from model import CharCNN
from textproc import CharVocab


def load_model(model_dir="models"):
    with open(f"{model_dir}/model_config.json") as f:
        cfg = json.load(f)
    vocab = CharVocab.load(f"{model_dir}/vocab.json")
    model = CharCNN(cfg["vocab_size"], cfg["emb_dim"], cfg["filters"], tuple(cfg["kernel_sizes"]), cfg["dropout"])
    model.load_state_dict(torch.load(f"{model_dir}/char_cnn.pt", map_location="cpu"))
    model.eval()                                          # dropout off
    return model, vocab, cfg


@torch.no_grad()
def profanity_probability(text, model, vocab, cfg):
    """P(comment contains profanity). Very long comments are read in overlapping windows."""
    ids = vocab.encode(text)
    n = cfg["max_chars"]
    starts = list(range(0, max(len(ids) - n, 0) + 1, n // 2))
    if starts[-1] + n < len(ids):
        starts.append(len(ids) - n)                       # make sure the end is covered
    windows = [ids[s:s + n] for s in starts]
    x = torch.zeros(len(windows), max(len(w) for w in windows), dtype=torch.long)
    for row, w in enumerate(windows):
        x[row, :len(w)] = torch.tensor(w)
    return torch.sigmoid(model(x)).max().item()           # the most profane window decides


def predict(texts, model=None, vocab=None, cfg=None):
    """Return a list of {text, probability, label} with label 1 = profanity, 0 = no profanity."""
    if model is None:
        model, vocab, cfg = load_model()
    results = []
    for text in texts:
        p = profanity_probability(text, model, vocab, cfg)
        results.append({"text": text, "probability": round(p, 4), "label": int(p >= cfg["threshold"])})
    return results


def show(result):
    word = "profanity" if result["label"] == 1 else "no profanity"
    print(f"{result['label']} ({word})  P(profanity) = {result['probability']:.4f}  |  {result['text']}")


if __name__ == "__main__":
    model, vocab, cfg = load_model()
    if len(sys.argv) > 1:
        for r in predict(sys.argv[1:], model, vocab, cfg):
            show(r)
    else:
        print("Type a comment and press Enter (empty line to stop).")
        while True:
            text = input("> ").strip()
            if not text:
                break
            show(predict([text], model, vocab, cfg)[0])
