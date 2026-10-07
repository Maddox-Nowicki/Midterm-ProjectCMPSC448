"""Required models: TextCNN and a bidirectional LSTM."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config as C


class TextCNN(nn.Module):
    """Kim-style CNN: embeddings -> parallel 1D convs (k=3,4,5) -> max-over-time -> FC."""
    def __init__(self, vocab_size, n_classes, emb_dim=C.EMB_DIM, n_filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lengths=None):
        e = self.emb(x).transpose(1, 2)                       # (B, E, L)
        pooled = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """Embeddings -> BiLSTM (packed, ignores padding) -> [max-pool ; mean-pool] -> FC."""
    def __init__(self, vocab_size, n_classes, emb_dim=C.EMB_DIM, hidden=128,
                 num_layers=1, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 4, n_classes)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lengths.clamp(min=1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(-1)
        maxp = out.masked_fill(~mask, -1e9).max(dim=1).values
        meanp = (out * mask).sum(1) / lengths.clamp(min=1).unsqueeze(1).float()
        return self.fc(self.drop(torch.cat([maxp, meanp], dim=1)))


def make_model(name, vocab_size, n_classes):
    return {"cnn": TextCNN, "lstm": BiLSTM}[name](vocab_size, n_classes)
