"""Step 4 (RQ4): what characteristics distinguish the LLMs?

Produces, in results/:
  rq4_feature_means.csv      mean of each handcrafted feature per LLM
  rq4_length_hist.png        response-length distributions
  rq4_ablation.csv           Random Forest accuracy with feature groups removed / alone
  rq4_feature_importance.csv permutation importance
  rq4_top_tokens.csv         most discriminative words/bigrams per LLM (TF-IDF + LogReg)
  rq4_nn_ablation.csv        (--nn) CNN with vs without formatting/case signals
"""
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import config as C
from data_utils import make_splits


def features(text):
    words = re.findall(r"\w+", text)
    nw = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    nc = max(len(text), 1)
    return dict(
        n_chars=len(text), n_words=len(words), n_sentences=len(sents),
        n_paragraphs=len(re.split(r"\n\s*\n", text.strip())),
        avg_word_len=sum(len(w) for w in words) / nw,
        type_token_ratio=len({w.lower() for w in words}) / nw,
        avg_sent_len=len(words) / max(len(sents), 1),
        n_newlines=text.count("\n"),
        n_bullets=len(re.findall(r"^\s*[-*\u2022]\s", text, re.M)),
        n_numbered=len(re.findall(r"^\s*\d+[.)]\s", text, re.M)),
        n_bold=text.count("**"), n_code=text.count("```"),
        n_headers=len(re.findall(r"^#+\s", text, re.M)),
        comma_rate=text.count(",") / nw, excl_rate=text.count("!") / nw,
        question_rate=text.count("?") / nw, colon_rate=text.count(":") / nw,
        uppercase_ratio=sum(c.isupper() for c in text) / nc,
        digit_ratio=sum(c.isdigit() for c in text) / nc,
    )


GROUPS = {
    "length": ["n_chars", "n_words", "n_sentences", "n_paragraphs"],
    "lexical": ["avg_word_len", "type_token_ratio", "avg_sent_len"],
    "formatting": ["n_newlines", "n_bullets", "n_numbered", "n_bold", "n_code", "n_headers"],
    "punctuation": ["comma_rate", "excl_rate", "question_rate", "colon_rate"],
    "char_mix": ["uppercase_ratio", "digit_ratio"],
}


def rf_acc(Xtr, ytr, Xte, yte, cols):
    clf = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xtr[cols], ytr)
    return accuracy_score(yte, clf.predict(Xte[cols]))


def main(run_nn):
    C.RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(C.DATA_PATH).fillna("")
    classes = sorted(df.llm_name.unique())
    tr, va, te = make_splits(df, seed=0)

    F = pd.DataFrame([features(t) for t in df["output"]], index=df.index)
    F["llm_name"] = df["llm_name"]
    means = F.groupby("llm_name").mean().round(3)
    means.to_csv(C.RESULTS / "rq4_feature_means.csv")
    print("\nFeature means per LLM:\n", means.T)

    fig, ax = plt.subplots(figsize=(6, 4))
    for c in classes:
        ax.hist(np.clip(F[F.llm_name == c].n_words, 0, 800), bins=40, alpha=0.5, label=c)
    ax.set_xlabel("Response length (words, clipped at 800)"); ax.set_ylabel("Count"); ax.legend()
    fig.tight_layout(); fig.savefig(C.RESULTS / "rq4_length_hist.png", dpi=150); plt.close(fig)

    # --- handcrafted features + random forest, with group ablation ---
    allc = [c for g in GROUPS.values() for c in g]
    Xtr, Xte = F.loc[tr.index, allc], F.loc[te.index, allc]
    ytr, yte = tr["llm_name"].values, te["llm_name"].values
    rows = [dict(setting="all features", acc=rf_acc(Xtr, ytr, Xte, yte, allc))]
    for g, cols in GROUPS.items():
        rest = [c for c in allc if c not in cols]
        rows.append(dict(setting=f"without {g}", acc=rf_acc(Xtr, ytr, Xte, yte, rest)))
    for g, cols in GROUPS.items():
        rows.append(dict(setting=f"only {g}", acc=rf_acc(Xtr, ytr, Xte, yte, cols)))
    abl = pd.DataFrame(rows).round(4)
    abl.to_csv(C.RESULTS / "rq4_ablation.csv", index=False)
    print("\nRandom-forest ablation (chance = %.3f):\n" % (1 / len(classes)), abl)

    rf = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0)
    imp = pd.DataFrame(dict(feature=allc, importance=pi.importances_mean)) \
            .sort_values("importance", ascending=False).round(4)
    imp.to_csv(C.RESULTS / "rq4_feature_importance.csv", index=False)
    print("\nPermutation importance:\n", imp.head(10))

    # --- most discriminative tokens ---
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), lowercase=False,
                          sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    Vtr = vec.fit_transform(tr["output"])
    lr = LogisticRegression(max_iter=300).fit(Vtr, ytr)
    print("\nTF-IDF + LogReg test acc: %.4f" % accuracy_score(yte, lr.predict(vec.transform(te["output"]))))
    names = np.array(vec.get_feature_names_out())
    top = []
    for i, c in enumerate(lr.classes_):
        for rank, j in enumerate(np.argsort(lr.coef_[i])[::-1][:15], 1):
            top.append(dict(llm=c, rank=rank, token=names[j], weight=round(lr.coef_[i][j], 3)))
    pd.DataFrame(top).to_csv(C.RESULTS / "rq4_top_tokens.csv", index=False)
    print(pd.DataFrame(top).groupby("llm").token.apply(lambda s: ", ".join(s)).to_string())

    # --- neural ablation: strip formatting/case, retrain CNN ---
    if run_nn:
        from train import run_experiment
        out = []
        for norm in (False, True):
            r = run_experiment(tr, va, te, "cnn", "output", classes, normalize=norm, seed=0)
            out.append(dict(setting="stripped (no markdown/newlines/case)" if norm else "original",
                            acc=r["acc"], macro_f1=r["macro_f1"]))
        o = pd.DataFrame(out).round(4)
        o.to_csv(C.RESULTS / "rq4_nn_ablation.csv", index=False)
        print("\nCNN signal-removal ablation:\n", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", action="store_true", help="also run CNN with stripped text")
    main(ap.parse_args().nn)
