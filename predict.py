"""Step 4: predict with the saved model (no retraining needed).

Run:  python predict.py "Керемет ойын, маған ұнады" "Мал ойын"    (one or more comments)
      python predict.py                                           (type comments one by one)
"""
import json
import sys

import torch

from model import build_model
from textproc import encode, load_vocab


def load_model(model_dir="models", name="cnn"):
    """name: "cnn" (the final model), "bilstm" or "transformer" (saved by compare_models.py)."""
    cfg = json.load(open(f"{model_dir}/model_config.json" if name == "cnn" else f"{model_dir}/char_{name}_config.json"))
    vocab = load_vocab(f"{model_dir}/vocab.json")
    model = build_model(name, cfg["vocab_size"], cfg)
    model.load_state_dict(torch.load(f"{model_dir}/char_{name}.pt", map_location="cpu"))
    model.eval()                                     # dropout off
    return model, vocab, cfg


@torch.no_grad()
def profanity_probability(text, model, vocab):
    """P(the comment contains profanity). The whole comment is read, whatever its length."""
    ids = torch.tensor([encode(text, vocab)])
    return torch.sigmoid(model(ids)).item()


def predict(texts, model=None, vocab=None, cfg=None):
    """Return [{text, probability, label}], label 1 = profanity, 0 = no profanity."""
    if model is None:
        model, vocab, cfg = load_model()
    results = []
    for text in texts:
        p = profanity_probability(text, model, vocab)
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
