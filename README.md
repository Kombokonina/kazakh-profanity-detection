# Kazakh Profanity Detection

A small character-level CNN, trained from scratch, reads one Kazakh comment and predicts **1 = profanity** or **0 = no profanity**, with the probability P(profanity).

| Test set (306 comments, threshold 0.50) | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Profanity class | **0.931** | **0.877** | **0.903** | 0.905 |

```bash
pip install -r requirements.txt
python predict.py "Мал ойын екен" "Керемет қосымша, маған ұнады"
# 1 (profanity)  P(profanity) = 0.9999  |  Мал ойын екен
# 0 (no profanity)  P(profanity) = 0.0022  |  Керемет қосымша, маған ұнады
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kombokonina/kazakh-profanity-detection/blob/main/notebooks/Kazakh_Profanity_Colab.ipynb) — the notebook predicts on your own text and can rerun everything.

## 1. Data

**Source.** All comments are real reviews from [KazSAnDRA](https://github.com/IS2AI/KazSAnDRA) (Yeshpanov & Varol, 2024; CC BY 4.0): reviews of apps and games, marketplace goods, places on maps and books. The bonus-project participants picked about 3,000 reviews with a list of swear words (a regular-expression search) and labelled them by hand. That shared file is [`data/kazakh_profanity_share.csv`](data/kazakh_profanity_share.csv): **3,061 comments**, 1,523 clean and 1,538 profane. It is in this repository, so no download is needed.

| Domain (from the id) | App store `pla` | Marketplace `kas`, `fli` | Maps `gis`, `map` | Books `kit` |
|---|---|---|---|---|
| Comments (profane) | 2,724 (1,520) | 191 (2) | 95 (10) | 51 (6) |

**No synthetic examples.** [`verify_source.py`](verify_source.py) finds all 3,061 ids in the original KazSAnDRA files: 3,002 texts are identical, and 59 differ only where personal details were masked.

**Privacy.** We replaced names, usernames, birth years, schools, a town and a home address in 53 comments with `<name>`, `@user`, `<year>`, `<school>`, `<place>`. [`data/privacy_masks.csv`](data/privacy_masks.csv) says where and what kind, never the original text.

### Labelling rules

**1 = profanity** if the comment has at least one of these:

- **1a** obscene or vulgar words (sex, genitals, excrement): қотақ, ам, сік-/сіг-, шешеңнің амы, көт, боқ;
- **1b** abusive insults at a person, group or the product: мал, есек, ит, чорт, лох, ақымақ;
- **1c** any spelling of these: masked (с\*ка, қот@қ), short (шшс, зб), stretched (Көтттт), misspelled (котак, зайбал), Latin (qotaq, pizdes);
- **1d** Russian swearing inside a Kazakh comment (говно, нахуй, сука, бля).

**0 = no profanity**: **2a** negative but clean («Жамаан ойын»); **2b** literal meaning (мал = livestock, ит = dog); **2c** talking about swearing without swearing; **2d** look-alike words (қотарылмайды = won't download, сикунд = секунд). Unclear cases (jokes, codes like «667») keep the group's label unless they clearly break these rules.

| Comment | Label | Rule |
|---|---|---|
| Мал ойын | 1 | 1b insult |
| Маған ұнайды бляя😎😎 | 1 | 1d: a positive review can still be profane |
| Б** з@еб@**л мына ойын | 1 | 1c masked |
| pizdes eken bargo 😁✊👍 | 1 | 1c Latin script |
| Жамаан ойын | 0 | 2a negative but clean |
| Маңғыстауда мал жоқ па? Дым шықпайды ғой | 0 | 2b мал = livestock |

Profanity is not negativity: 590 profane comments are 5-star reviews, and 538 clean ones are 1-star reviews.

### Label checks

The XLSX and CSV versions of the shared file disagreed on 5 labels; the XLSX was right in 4, so we use it. [`check_labels.py`](check_labels.py) then lists (1) comments labelled 0 that contain an obscene word, from all splits, and (2) the 60 train/val comments where a simple TF-IDF + logistic regression model, with 5-fold cross-validation, disagrees most with the label. The test set is not part of (2). We read them all and fixed **7 clear errors**, each with a reason in [`data/label_fixes.csv`](data/label_fixes.csv). One error was found too late: `pla012970pla` («…заибал бан…») should be 1 but is 0; we left it so the data matches the saved model.

### Splits

[`prepare_data.py`](prepare_data.py), seed **5113**:

1. Removes exact duplicates (only case or spaces differ): 3,061 → 3,059.
2. Groups near-duplicates: comments whose 3-letter pieces overlap by at least 80% (e.g. «Қотақбас» and «Қотақбассссс»): 14 groups, 30 comments.
3. Splits by group with `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=5113)`: 1 part test, 1 part validation, 8 parts train. A group never crosses splits.

| Split | Comments | Clean (0) | Profane (1) |
|---|---|---|---|
| train | 2,447 | 1,218 | 1,229 |
| val | 306 | 152 | 154 |
| test | 306 | 152 | 154 |

We did not reuse the shared file's own split: it had no seed and the same review in both train and val. The character vocabulary is built from the training set only.

## 2. Model

**Why a character CNN.** Swear words are short letter patterns written in many ways (с\*ка, Көтттт, qotaq, Russian words). A word model sees most of these as unknown words; a character model sees all of them. Each convolution filter learns one short pattern, and max pooling asks "did it appear anywhere?". The model has 105,649 parameters and trains in about 2 minutes on a CPU. On validation it beats a simple TF-IDF + logistic regression baseline (F1 0.931 vs 0.906).

**One comment, from text to prediction** («МАЛ ОЙЫН!!!!!»):

1. Normalize ([`textproc.py`](textproc.py)): lowercase, 3+ repeated characters → 2 → «мал ойын!!». Punctuation and emoji stay, because `*` and `@` often hide swear words.
2. Characters → ids from [`models/vocab.json`](models/vocab.json): `[92, 80, 91, 2, 94, 89, 107, 93, 3, 3]`. A character not seen at least twice in training becomes 1 (unknown); 0 is padding.
3. Embedding ([`model.py`](model.py)): each id → 48 learned numbers, shape (1, 10, 48).
4. Convolutions: 128 filters each of width 3, 5 and 7 characters, with ReLU → (1, 128, 10) each.
5. Max pooling: each filter keeps its best score in the comment → 384 numbers, whatever the length.
6. Dropout 0.5 (training only), then Linear(384 → 1) → logit 8.33 → sigmoid → P = 0.9998.
7. Threshold 0.50 → label **1**.

**Long comments.** Training cuts comments at 1,024 characters (only 1 of 2,447 is longer). Prediction reads the whole comment, because max pooling works for any length.

**Settings** ([`config.json`](config.json)): embedding 48, 128 filters per width (3, 5, 7), dropout 0.5, `BCEWithLogitsLoss`, Adam with learning rate 0.001, batch 32, up to 30 epochs, seed 5113. We keep the epoch with the best validation F1 and stop after 6 epochs without improvement (best: epoch **15** of 21). The threshold is the value from 0.05–0.95 with the best validation F1: **0.50**. Validation: precision 0.946, recall 0.916, F1 0.931. Training is deterministic: rerunning everything gives identical files.

## 3. Results

The test set was used only after the epoch and threshold were fixed ([`evaluate.py`](evaluate.py)).

![Confusion matrix](results/confusion_matrix.png)

142 clean and 135 profane comments are right; there are 10 false positives and 19 false negatives. The 95% bootstrap interval for F1 is 0.864–0.935, so small differences between models mean little. All 154 profane test comments are app reviews. The other 34 test comments (marketplace, maps, books) are clean, and 33 were classified correctly. Every test prediction is in [`results/test_predictions.csv`](results/test_predictions.csv).

**Moderation.** A false positive hides an honest review; a false negative leaves swearing visible, also to children reading game reviews. If flagged comments are deleted automatically, use a high threshold; if a human checks them, use a low one. Validation: threshold 0.3 → precision 0.900, recall 0.935; 0.9 → precision 0.992, recall 0.792. A sensible policy: hide above 0.9, send 0.3–0.9 to a moderator, publish below 0.3.

### Error analysis (test set)

| Comment | True | P | Why |
|---|---|---|---|
| «өте күшті емес **сука**» | 1 | 0.013 | One swear word among positive words. In training, «сука» almost always comes with other swear words. Without «күшті», P = 0.38. |
| «**Ам** сияқты ұнады👍» | 1 | 0.008 | «ам» has 2 letters and hides inside clean words (адам); «ұнады» (liked) pulls the score down. |
| «Wws **amshyk** samokat **ebanuty** …» | 1 | 0.115 | Latin script: only about 75 training comments are mainly Latin. |
| «Неге **қотар**лмайды» (won't download) | 0 | 0.975 | Looks like «қотақ» (297 training comments, all profane); the clean verb «қотару» never appears in training. |
| «Маңғыстауда **мал** жоқ па? …» | 0 | 0.989 | Literal «мал» (livestock): 92% of comments with the word «мал» are profane. «Мал шаруашылығы» is handled (13 clean training examples), other literal uses are not. |
| «Супер Боди (&68587 **\*^\*^**…» | 0 | 0.746 | `*` usually hides a swear word (43 of 51 training comments with `*` are profane). |

Single words are also risky: «Кітап» (book) gets 0.95, because most very short training comments are profane.

**Improvements:** (1) more clean comments with look-alike and literal words (қотару, мал, көк) and more comments with only one swear word; (2) careful augmentation — Latin-script copies and random `*` masking, but not swear-word fragments, which can make words like «су» look profane (from «сука»); (3) a second annotator for unclear words (далбаса, боқ, 667); (4) fine-tuning a pretrained Kazakh or multilingual transformer, which reads context.

**Sample predictions:**

```text
1 (profanity)     P = 0.9950 | Көшірінбейд виртка кірейін десем мал мал телеграм плей маркет мал   (test, correct)
0 (no profanity)  P = 0.0283 | МАГАН тик ток унады😊                                              (test, correct)
0 (no profanity)  P = 0.1695 | Мал шаруашылығына арналған жақсы кітап                             (new text)
0 (no profanity)  P = 0.4994 | с*ка не ашылмайды                                                   (new text, a near miss)
```

**Transparency.** We trained and tested three times with the same code, settings and seed. Each new run only followed more privacy masking (after run 1, then after run 2). Test F1 was 0.905, 0.901 and 0.903. We report run 3, the model in this repository.

## 4. How to run

Python 3.12 or newer:

```bash
git clone https://github.com/Kombokonina/kazakh-profanity-detection.git
cd kazakh-profanity-detection
pip install -r requirements.txt

python predict.py "Мал ойын екен"     # predict (no training needed); with no text it asks you to type
python prepare_data.py                 # 1. duplicates, groups, split, label fixes -> data/processed/
python train.py                        # 2. train, pick epoch and threshold on validation -> models/, results/
python evaluate.py                     # 3. test metrics, confusion matrix, predictions -> results/
python check_labels.py                 # optional: label checks and the baseline
```

In Python: `from predict import predict; predict(["Мал ойын екен"])` returns `[{'text': 'Мал ойын екен', 'probability': 0.9999, 'label': 1}]`. Exact versions are in [`requirements.txt`](requirements.txt). In Colab the preinstalled packages are enough.

## 5. Limitations

- 89% of the comments are app and game reviews; only 2 of 191 marketplace reviews are profane, so real product reviews are barely tested.
- The labels come from students, one label per comment, with no agreement check. Mild words (далбаса, боқ, ит) are partly subjective.
- The model sees letters, not meaning: it confuses literal and insulting «мал», look-alikes (қотару, көк) and misses new spellings or codes (гондон, 667).
- 3,059 comments, one seed, 306 test comments: test F1 is known only to about ±0.04.
- We masked every personal detail our checks found, but a missed name may remain.

## 6. License and citation

Code: MIT License ([`LICENSE`](LICENSE)). Data: KazSAnDRA, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); labels by the DSSC5113 bonus-project participants.

> Yeshpanov, R., & Varol, H. A. (2024). KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes. *LREC-COLING 2024.* [arXiv:2403.19335](https://arxiv.org/abs/2403.19335)
