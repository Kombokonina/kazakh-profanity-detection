"""Optional: compare the character CNN with a BiLSTM, a small Transformer and the simple baseline.

Run:  python compare_models.py                  (after prepare_data.py; about 30 minutes on a CPU)
      python compare_models.py transformer      (only some models; the other rows of the table stay)
Every neural model gets the same data, vocabulary, training loop, settings and seed.
The model is chosen on validation. The test set is scored at the end, for reference.
Writes results/model_comparison.csv (one row per model, saved as soon as the model is done)
and saves the BiLSTM and the Transformer in models/.
"""
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from check_labels import baseline
from model import build_model
from textproc import build_vocab, encode
from train import make_batches, predict_logits

BASELINE = "TF-IDF + logistic regression (baseline)"
NAMES = {"cnn": "Character CNN (final)", "bilstm": "Character BiLSTM", "transformer": "Character Transformer"}
TABLE = "results/model_comparison.csv"
EXAMPLES = ["өте күшті емес сука", "Неге қотарлмайды", "Маңғыстауда мал жоқ па? Дым шықпайды ғой", "Кітап",
            "Мал ойын екен", "Керемет қосымша, маған ұнады"]


def fit(model, train_ids, y_train, val_ids, y_val, cfg):
    """The same loop as train.py: keep the epoch with the best validation F1, stop after `patience` worse epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    shuffler = torch.Generator().manual_seed(cfg["seed"])
    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        for x, y in make_batches(train_ids, y_train, cfg["batch_size"], shuffle=True, generator=shuffler):
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        val_f1 = f1_score(y_val, (torch.sigmoid(predict_logits(model, val_ids)).numpy() >= 0.5).astype(int))
        print(f"  epoch {epoch:2d}  validation F1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return best_epoch


def result(name, params, seconds, best_epoch, y, val_prob, test_prob):
    """Threshold with the best validation F1 (0.05 to 0.95), then the scores at that threshold."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    threshold = float(thresholds[np.argmax([f1_score(y["val"], (val_prob >= t).astype(int)) for t in thresholds])])
    p = (test_prob >= threshold).astype(int)
    return {"model": name, "parameters": params, "train_seconds": round(seconds), "best_epoch": best_epoch,
            "threshold": threshold, "val_f1": round(f1_score(y["val"], (val_prob >= threshold).astype(int)), 4),
            "test_precision": round(precision_score(y["test"], p), 4), "test_recall": round(recall_score(y["test"], p), 4),
            "test_f1": round(f1_score(y["test"], p), 4), "test_accuracy": round(accuracy_score(y["test"], p), 4)}


def save(rows):
    order = [BASELINE] + list(NAMES.values())
    table = pd.DataFrame([rows[m] for m in order if m in rows])
    table.astype({"parameters": "Int64", "best_epoch": "Int64"}).to_csv(TABLE, index=False)


def main():
    names = sys.argv[1:] or list(NAMES)
    cfg = json.load(open("config.json"))
    data = {s: pd.read_csv(f"data/processed/{s}.csv") for s in ["train", "val", "test"]}
    y = {s: d["label"].to_numpy() for s, d in data.items()}
    rows = {r["model"]: r for r in pd.read_csv(TABLE).to_dict("records")} if os.path.exists(TABLE) else {}

    start = time.time()
    lr = baseline().fit(data["train"]["text"], y["train"])
    weights = lr[-1].coef_.size + 1                  # one weight per letter chunk, plus the bias
    rows[BASELINE] = result(BASELINE, weights, time.time() - start, None, y,
                            lr.predict_proba(data["val"]["text"])[:, 1], lr.predict_proba(data["test"]["text"])[:, 1])
    save(rows)

    vocab = build_vocab(data["train"]["text"], cfg["min_char_count"])         # the same vocabulary as train.py
    vocab_size = len(vocab) + 2
    ids = {s: [encode(t, vocab, cfg["max_chars"]) for t in d["text"]] for s, d in data.items()}
    examples = {"comment": EXAMPLES}
    for name in names:
        print(NAMES[name], flush=True)
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        model = build_model(name, vocab_size, cfg)
        params = sum(p.numel() for p in model.parameters())
        start = time.time()
        best_epoch = fit(model, ids["train"], y["train"].tolist(), ids["val"], y["val"], cfg)
        seconds = time.time() - start
        val_prob, test_prob = (torch.sigmoid(predict_logits(model, ids[s])).numpy() for s in ["val", "test"])
        rows[NAMES[name]] = result(NAMES[name], params, seconds, best_epoch, y, val_prob, test_prob)
        save(rows)
        print(rows[NAMES[name]], flush=True)
        examples[name] = torch.sigmoid(predict_logits(model, [encode(t, vocab) for t in EXAMPLES])).numpy().round(4)
        if name != "cnn":                            # the CNN is saved by train.py
            torch.save(model.state_dict(), f"models/char_{name}.pt")
            keep = ["emb_dim", "dropout", "max_chars"] + [k for k in cfg if k.startswith(("lstm", "transformer"))]
            json.dump({**{k: cfg[k] for k in keep}, "vocab_size": vocab_size, "threshold": rows[NAMES[name]]["threshold"],
                       "best_epoch": best_epoch, "seed": cfg["seed"], "trainable_parameters": params},
                      open(f"models/char_{name}_config.json", "w"), indent=2)

    print(pd.read_csv(TABLE).to_string(index=False))
    print("\nP(profanity) for a few comments:")
    print(pd.DataFrame(examples).to_string(index=False))


if __name__ == "__main__":
    main()
