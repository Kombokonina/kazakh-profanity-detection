"""Step 2: train the character CNN, then pick the best epoch and the threshold on validation.

Run:  python train.py
Saves the model to models/ and the training log, validation scores and curves to results/.
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
from textproc import build_vocab, encode, save_vocab


def make_batches(ids, labels, batch_size, shuffle=False, generator=None):
    """Yield (x, y). Shorter comments are padded with 0 to the longest one in the batch."""
    order = torch.randperm(len(ids), generator=generator).tolist() if shuffle else list(range(len(ids)))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        x = torch.zeros(len(chunk), max(len(ids[i]) for i in chunk), dtype=torch.long)
        for row, i in enumerate(chunk):
            x[row, :len(ids[i])] = torch.tensor(ids[i])
        yield x, torch.tensor([labels[i] for i in chunk], dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, ids):
    model.eval()                                     # dropout off
    return torch.cat([model(x) for x, _ in make_batches(ids, [0] * len(ids), 128)])


def scores(y_true, y_pred):
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def main():
    cfg = json.load(open("config.json"))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")

    vocab = build_vocab(train["text"], cfg["min_char_count"])          # training data only
    save_vocab(vocab, "models/vocab.json")
    vocab_size = len(vocab) + 2                                         # + padding and unknown
    train_ids = [encode(t, vocab, cfg["max_chars"]) for t in train["text"]]
    val_ids = [encode(t, vocab, cfg["max_chars"]) for t in val["text"]]
    y_train, y_val = train["label"].tolist(), val["label"].to_numpy()

    model = CharCNN(vocab_size, cfg["emb_dim"], cfg["filters"], cfg["kernel_sizes"], cfg["dropout"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocabulary: {vocab_size} ids | trainable parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    shuffler = torch.Generator().manual_seed(cfg["seed"])

    log, best_f1, best_epoch, best_state = [], -1.0, 0, None
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
        if val_f1 > best_f1:                         # keep the epoch with the best validation F1
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg["patience"]:  # early stopping
            print(f"stopping: no better validation F1 for {cfg['patience']} epochs")
            break

    # Best epoch back, then the threshold with the best validation F1
    model.load_state_dict(best_state)
    val_prob = torch.sigmoid(predict_logits(model, val_ids)).numpy()
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1s = [f1_score(y_val, (val_prob >= t).astype(int)) for t in thresholds]
    threshold = float(thresholds[np.argmax(f1s)])
    print(f"best epoch {best_epoch} | chosen threshold {threshold:.2f}")

    torch.save(model.state_dict(), "models/char_cnn.pt")
    model_config = {k: cfg[k] for k in ["emb_dim", "filters", "kernel_sizes", "dropout", "max_chars"]}
    model_config.update({"vocab_size": vocab_size, "threshold": threshold, "best_epoch": best_epoch,
                         "seed": cfg["seed"], "trainable_parameters": n_params})
    json.dump(model_config, open("models/model_config.json", "w"), indent=2)

    pd.DataFrame(log).to_csv("results/training_log.csv", index=False)
    validation = {"best_epoch": best_epoch, "threshold": threshold,
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
    ax[1].axvline(best_epoch, color="gray", linestyle=":", label=f"chosen epoch {best_epoch}")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation F1 (profanity, threshold 0.5)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig("results/training_curves.png", dpi=150)
    print("saved model, vocabulary, config and training results")


if __name__ == "__main__":
    main()
