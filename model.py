"""The classifier: a character-level CNN."""
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
