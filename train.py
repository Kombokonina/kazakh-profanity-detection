"""Step 2: train the character CNN; pick the best epoch and the decision threshold on validation.

Run:  python train.py
Writes models/char_cnn.pt, models/vocab.json, models/model_config.json and
results/training_log.csv, results/validation_metrics.json, results/training_curves.png.
The test set is not used here.
"""
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from model import CharCNN
from textproc import CharVocab


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(ids, labels, batch_size, shuffle=False, generator=None):
    """Yield (x, y): x is (batch, longest comment in batch) padded with 0."""
    order = torch.randperm(len(ids), generator=generator).tolist() if shuffle else list(range(len(ids)))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        x = torch.zeros(len(chunk), max(len(ids[i]) for i in chunk), dtype=torch.long)
        for row, i in enumerate(chunk):
            x[row, :len(ids[i])] = torch.tensor(ids[i])
        yield x, torch.tensor([labels[i] for i in chunk], dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, ids, batch_size=128):
    model.eval()                                          # dropout off
    return torch.cat([model(x) for x, _ in make_batches(ids, [0] * len(ids), batch_size)])


def scores(y_true, y_pred):
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def main():
    cfg = json.load(open("config.json"))
    set_seed(cfg["seed"])
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")

    vocab = CharVocab.build(train["text"], cfg["min_char_count"])        # training data only
    vocab.save("models/vocab.json")
    train_ids = [vocab.encode(t, cfg["max_chars"]) for t in train["text"]]
    val_ids = [vocab.encode(t, cfg["max_chars"]) for t in val["text"]]
    y_train, y_val = train["label"].tolist(), val["label"].to_numpy()

    model = CharCNN(len(vocab), cfg["emb_dim"], cfg["filters"], tuple(cfg["kernel_sizes"]), cfg["dropout"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"vocabulary: {len(vocab)} ids | trainable parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    shuffler = torch.Generator().manual_seed(cfg["seed"])

    log, best = [], {"f1": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        total = 0.0
        for x, y in make_batches(train_ids, y_train, cfg["batch_size"], shuffle=True, generator=shuffler):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        val_logits = predict_logits(model, val_ids)
        val_loss = loss_fn(val_logits, torch.tensor(y_val, dtype=torch.float32)).item()
        val_f1 = f1_score(y_val, (torch.sigmoid(val_logits).numpy() >= 0.5).astype(int))
        log.append({"epoch": epoch, "train_loss": total / len(y_train), "val_loss": val_loss, "val_f1": val_f1})
        print(f"epoch {epoch:2d}  train loss {total / len(y_train):.4f}  val loss {val_loss:.4f}  val F1 {val_f1:.4f}")
        if val_f1 > best["f1"] or (val_f1 == best["f1"] and val_loss < best["loss"]):
            best = {"f1": val_f1, "loss": val_loss, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= cfg["patience"]:
            print(f"stopping: no better validation F1 for {cfg['patience']} epochs")
            break

    # Keep the best epoch, then choose the threshold that gives the best validation F1
    model.load_state_dict(best["state"])
    val_prob = torch.sigmoid(predict_logits(model, val_ids)).numpy()
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1s = [f1_score(y_val, (val_prob >= t).astype(int)) for t in thresholds]
    threshold = float(max(zip(f1s, -abs(thresholds - 0.5), thresholds))[2])   # ties -> closest to 0.5
    print(f"best epoch {best['epoch']} | chosen threshold {threshold:.2f}")

    torch.save(model.state_dict(), "models/char_cnn.pt")
    model_config = {k: cfg[k] for k in ["emb_dim", "filters", "kernel_sizes", "dropout", "max_chars"]}
    model_config.update({"vocab_size": len(vocab), "threshold": threshold, "best_epoch": best["epoch"],
                         "seed": cfg["seed"], "trainable_parameters": n_params})
    json.dump(model_config, open("models/model_config.json", "w"), indent=2)

    pd.DataFrame(log).to_csv("results/training_log.csv", index=False)
    validation = {"best_epoch": best["epoch"], "threshold": threshold,
                  "at_0.5": scores(y_val, (val_prob >= 0.5).astype(int)),
                  "at_threshold": scores(y_val, (val_prob >= threshold).astype(int)),
                  "threshold_search": {f"{t:.2f}": round(f, 4) for t, f in zip(thresholds, f1s)}}
    json.dump(validation, open("results/validation_metrics.json", "w"), indent=2)

    epochs = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, [r["train_loss"] for r in log], "o-", label="train loss")
    ax[0].plot(epochs, [r["val_loss"] for r in log], "s-", label="validation loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("binary cross-entropy"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(epochs, [r["val_f1"] for r in log], "o-", color="tab:green")
    ax[1].axvline(best["epoch"], color="gray", linestyle=":", label=f"chosen epoch {best['epoch']}")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation F1 (profanity, threshold 0.5)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig("results/training_curves.png", dpi=150)
    print("saved model, vocabulary, config and training results")


if __name__ == "__main__":
    main()
