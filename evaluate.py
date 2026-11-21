"""Step 3: evaluate the saved model once on the held-out test set.

Run:  python evaluate.py
Writes results/test_metrics.json, results/confusion_matrix.png and results/test_predictions.csv.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from predict import load_model, profanity_probability


def main():
    model, vocab, cfg = load_model()
    test = pd.read_csv("data/processed/test.csv")
    test["probability"] = [round(profanity_probability(t, model, vocab, cfg), 4) for t in test["text"]]
    test["prediction"] = (test["probability"] >= cfg["threshold"]).astype(int)
    test["correct"] = test["prediction"] == test["label"]
    y, p = test["label"], test["prediction"]

    cm = confusion_matrix(y, p, labels=[0, 1])
    metrics = {
        "test_examples": len(test),
        "threshold": cfg["threshold"],
        "precision_profanity": round(precision_score(y, p), 4),
        "recall_profanity": round(recall_score(y, p), 4),
        "f1_profanity": round(f1_score(y, p), 4),
        "accuracy": round(accuracy_score(y, p), 4),
        "confusion_matrix": {"true_0_pred_0": int(cm[0, 0]), "true_0_pred_1": int(cm[0, 1]),
                             "true_1_pred_0": int(cm[1, 0]), "true_1_pred_1": int(cm[1, 1])},
        "by_source": {src: {"examples": len(g), "profane": int(g["label"].sum()),
                            "correct": int(g["correct"].sum())}
                      for src, g in test.groupby("source")},
    }
    with open("results/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    test[["kazsandra_id", "source", "text", "label", "probability", "prediction", "correct"]].to_csv(
        "results/test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["0 no profanity", "1 profanity"]); ax.set_yticks([0, 1], ["0 no profanity", "1 profanity"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true label")
    ax.set_title(f"Test set ({len(test)} comments, threshold {cfg['threshold']:.2f})")
    fig.tight_layout(); fig.savefig("results/confusion_matrix.png", dpi=150)

    print(json.dumps({k: v for k, v in metrics.items() if k != "by_source"}, indent=2))
    for name, rows in [("FALSE POSITIVES (clean, predicted profane)", test[(y == 0) & (p == 1)]),
                       ("FALSE NEGATIVES (profane, predicted clean)", test[(y == 1) & (p == 0)])]:
        print(f"\n{name}: {len(rows)}")
        for _, r in rows.iterrows():
            print(f"  P={r['probability']:.3f}  {r['text'][:150]}")


if __name__ == "__main__":
    main()
