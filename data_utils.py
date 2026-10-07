"""Tokenization, vocabulary, encoding, and prompt-grouped splitting."""
import re
from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
import config as C

TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")   # words, punctuation, newlines kept as tokens


def strip_signals(text):
    """Remove markdown symbols/newlines and lowercase (used in the RQ4 ablation)."""
    text = re.sub(r"[*#`_>|~\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def tokenize(text, normalize=False):
    if normalize:
        text = strip_signals(text)
    return ["<nl>" if t == "\n" else t for t in TOKEN_RE.findall(text)]


def build_tokens(df, mode, normalize=False):
    """mode: 'output' | 'input' | 'both'."""
    out = []
    for inp, resp in zip(df["input"], df["output"]):
        if mode == "output":
            t = tokenize(resp, normalize)[:C.MAX_OUT]
        elif mode == "input":
            t = tokenize(inp, normalize)[:C.MAX_IN]
        elif mode == "both":
            t = (tokenize(inp, normalize)[:C.MAX_IN] + ["<sep>"]
                 + tokenize(resp, normalize)[:C.MAX_OUT])
        else:
            raise ValueError(mode)
        out.append(t)
    return out


def seq_len(mode):
    return {"output": C.MAX_OUT, "input": C.MAX_IN,
            "both": C.MAX_IN + 1 + C.MAX_OUT}[mode]


def build_vocab(token_lists):
    """Vocabulary from TRAINING data only. 0 = <pad>, 1 = <unk>."""
    cnt = Counter(t for ts in token_lists for t in ts)
    itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common(C.MAX_VOCAB) if c >= C.MIN_FREQ]
    return {w: i for i, w in enumerate(itos)}


def encode(token_lists, vocab, max_len):
    X = np.zeros((len(token_lists), max_len), dtype=np.int64)
    for i, ts in enumerate(token_lists):
        ids = [vocab.get(t, 1) for t in ts[:max_len]]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)


def make_splits(df, seed, val=0.1, test=0.1):
    """80/10/10 split GROUPED BY PROMPT, so the same prompt never appears in
    both train and test (several LLMs answer the same prompt)."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=val + test, random_state=seed)
    tr_idx, rest_idx = next(g1.split(df, groups=df["prompt_id"].values))
    rest = df.iloc[rest_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=test / (val + test), random_state=seed)
    v_idx, t_idx = next(g2.split(rest, groups=rest["prompt_id"].values))
    return df.iloc[tr_idx], rest.iloc[v_idx], rest.iloc[t_idx]


def split_train_val(df, seed, val=0.1):
    g = GroupShuffleSplit(n_splits=1, test_size=val, random_state=seed)
    tr_idx, v_idx = next(g.split(df, groups=df["prompt_id"].values))
    return df.iloc[tr_idx], df.iloc[v_idx]
