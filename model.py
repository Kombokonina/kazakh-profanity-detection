"""The classifiers: a character-level CNN (the final model), a BiLSTM and a small Transformer.
All three read character ids and end the same way: max over the comment -> dropout -> 1 logit."""
import math

import torch
import torch.nn as nn


class CharCNN(nn.Module):
    """Character ids -> embeddings -> 3 convolutions -> max over the comment -> 1 logit."""

    def __init__(self, vocab_size, emb_dim=48, filters=128, kernel_sizes=(3, 5, 7), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, filters, k, padding="same") for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(kernel_sizes), 1)

    def forward(self, ids):                          # ids: (batch, length), 0 = padding
        real = (ids != 0).unsqueeze(1)               # True where there is a real character
        x = self.embedding(ids).transpose(1, 2)      # (batch, emb_dim, length)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x)).masked_fill(~real, 0.0)   # pattern scores; padding gets 0
            pooled.append(h.max(dim=2).values)       # best score of each filter in the comment
        features = self.dropout(torch.cat(pooled, dim=1))
        return self.out(features).squeeze(1)         # logit; sigmoid(logit) = P(profanity)


class CharBiLSTM(nn.Module):
    """Character ids -> embeddings -> LSTM left-to-right and right-to-left -> max over the comment -> 1 logit."""

    def __init__(self, vocab_size, emb_dim=48, hidden=80, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, ids):
        real = ids != 0
        packed = nn.utils.rnn.pack_padded_sequence(            # the LSTM skips the padding
            self.embedding(ids), real.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.size(1))
        h = h.masked_fill(~real.unsqueeze(2), float("-inf"))   # (batch, length, 2 * hidden)
        return self.out(self.dropout(h.max(dim=1).values)).squeeze(1)


class CharTransformer(nn.Module):
    """Character ids -> embeddings + position codes -> self-attention layers -> max over the comment -> 1 logit."""

    def __init__(self, vocab_size, emb_dim=48, heads=4, layers=2, ff_dim=384, attn_dropout=0.1, dropout=0.5,
                 max_chars=256):
        super().__init__()
        self.max_chars = max_chars
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(emb_dim, heads, ff_dim, attn_dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(emb_dim), enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, 1)

    def forward(self, ids):
        ids = ids[:, :self.max_chars]                # the first 256 characters: attention cost grows with length²
        pad = ids == 0                               # True = padding, attention ignores it
        x = self.embedding(ids) + positions(ids.size(1), self.embedding.embedding_dim)
        h = self.encoder(x, src_key_padding_mask=pad)
        h = h.masked_fill(pad.unsqueeze(2), float("-inf"))
        return self.out(self.dropout(h.max(dim=1).values)).squeeze(1)


def positions(length, dim):
    """Fixed sine and cosine codes, one per position, so attention knows the order of the letters."""
    pos = torch.arange(length).unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    codes = torch.zeros(length, dim)
    codes[:, 0::2] = torch.sin(pos * freq)
    codes[:, 1::2] = torch.cos(pos * freq)
    return codes


def build_model(name, vocab_size, cfg):
    if name == "cnn":
        return CharCNN(vocab_size, cfg["emb_dim"], cfg["filters"], cfg["kernel_sizes"], cfg["dropout"])
    if name == "bilstm":
        return CharBiLSTM(vocab_size, cfg["emb_dim"], cfg["lstm_hidden"], cfg["dropout"])
    if name == "transformer":
        return CharTransformer(vocab_size, cfg["emb_dim"], cfg["transformer_heads"], cfg["transformer_layers"],
                               cfg["transformer_ff"], cfg["transformer_dropout"], cfg["dropout"],
                               cfg["transformer_max_chars"])
    raise ValueError(f"unknown model: {name}")
