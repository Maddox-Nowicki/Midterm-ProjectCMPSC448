"""Train + evaluate one model on one input configuration."""
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import config as C
from data_utils import build_tokens, build_vocab, encode, seq_len
from models import make_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _loader(df, tokens, vocab, mode, cid, shuffle):
    X = encode(tokens, vocab, seq_len(mode))
    lengths = (X != 0).sum(1)
    y = torch.tensor(df["llm_name"].map(cid).values, dtype=torch.long)
    return DataLoader(TensorDataset(X, lengths, y), batch_size=C.BATCH, shuffle=shuffle)


def _predict(model, loader):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, l, y in loader:
            P.append(model(x, l).argmax(1)); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def run_experiment(train_df, val_df, test_df, model_name, mode, classes,
                   normalize=False, seed=0, epochs=None, verbose=True):
    """model_name: 'cnn'|'lstm'; mode: 'output'|'input'|'both'."""
    epochs = epochs or C.EPOCHS
    set_seed(seed)
    cid = {c: i for i, c in enumerate(classes)}
    tok = {k: build_tokens(d, mode, normalize)
           for k, d in (("tr", train_df), ("va", val_df), ("te", test_df))}
    vocab = build_vocab(tok["tr"])
    tr = _loader(train_df, tok["tr"], vocab, mode, cid, True)
    va = _loader(val_df, tok["va"], vocab, mode, cid, False)
    te = _loader(test_df, tok["te"], vocab, mode, cid, False)

    model = make_model(model_name, len(vocab), len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, best_ep, bad, history = -1, None, 0, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); tot = n = 0
        for x, l, y in tr:
            opt.zero_grad()
            loss = loss_fn(model(x, l), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        vp, vy = _predict(model, va)
        vf1 = f1_score(vy, vp, average="macro")
        history.append(dict(epoch=ep, train_loss=tot / n,
                            val_acc=float(accuracy_score(vy, vp)), val_f1=float(vf1)))
        if verbose:
            print(f"    [{model_name}/{mode}] ep{ep} loss={tot/n:.3f} "
                  f"val_acc={history[-1]['val_acc']:.3f} val_f1={vf1:.3f} ({time.time()-t0:.0f}s)")
        if vf1 > best_f1:
            best_f1, best_state, best_ep, bad = vf1, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break
    model.load_state_dict(best_state)

    tp, ty = _predict(model, te)
    cm = confusion_matrix(ty, tp, labels=range(len(classes)))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    per_domain = {}
    if "domain" in test_df.columns:
        per_domain = pd.Series(tp == ty).groupby(test_df["domain"].values).mean().to_dict()
    return dict(model=model_name, mode=mode, normalize=normalize, seed=seed,
                acc=float(accuracy_score(ty, tp)),
                macro_f1=float(f1_score(ty, tp, average="macro")),
                confusion=cm.tolist(), classes=list(classes),
                per_class_recall=dict(zip(classes, recall.tolist())),
                per_domain_acc=per_domain, best_epoch=best_ep, history=history,
                n_train=len(train_df), n_test=len(test_df), vocab_size=len(vocab))
