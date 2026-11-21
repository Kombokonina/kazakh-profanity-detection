"""The classifier: a character-level CNN."""
import torch
import torch.nn as nn


class CharCNN(nn.Module):
    """Characters -> embeddings -> 3 convolutions -> max over the comment -> 1 logit.

    Each convolution filter slides over the comment and learns to fire on a short
    character pattern (3, 5 or 7 characters), for example 'қот', 'сік' or 'нах'.
    Max pooling keeps the strongest match anywhere in the comment, so it works for
    any comment length and a swear word counts the same at the start or the end.
    """

    def __init__(self, vocab_size, emb_dim=48, filters=128, kernel_sizes=(3, 5, 7), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, filters, k, padding="same") for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(kernel_sizes), 1)

    def forward(self, ids):                              # ids: (B, T) character ids, 0 = padding
        real = (ids != 0).unsqueeze(1)                   # (B, 1, T) True at real characters
        x = self.embedding(ids).transpose(1, 2)          # (B, E, T)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))                      # (B, F, T) filter scores at each position
            h = h.masked_fill(~real, 0.0)                # ignore padding positions
            pooled.append(h.max(dim=2).values)           # (B, F) strongest match per filter
        features = self.dropout(torch.cat(pooled, dim=1))   # (B, 3F)
        return self.out(features).squeeze(1)             # (B,) logit; sigmoid(logit) = P(profane)
