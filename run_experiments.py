"""Steps 2-3: RQ1/RQ2 (input-only vs output-only vs both) and RQ3 (domain shift).

  python run_experiments.py --rq 12          # RQ1 + RQ2
  python run_experiments.py --rq 3           # RQ3 (run after RQ1/2 for the baseline column)
  python run_experiments.py --rq all
  python run_experiments.py --rq all --quick # tiny smoke test (minutes, not hours)
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from data_utils import make_splits, split_train_val
from train import run_experiment

MODELS = ["cnn", "lstm"]
MODES = ["output", "input", "both"]


def load(quick):
    df = pd.read_csv(C.DATA_PATH).fillna("")
    if quick:
        df = pd.concat([g.sample(min(len(g), 400), random_state=0)
                        for _, g in df.groupby("llm_name")])
    return df, sorted(df["llm_name"].unique())


def plot_confusion(res, path):
    cm = np.array(res["confusion"]); cls = res["classes"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=45, ha="right")
    ax.set_yticks(range(len(cls))); ax.set_yticklabels(cls)
    for i in range(len(cls)):
        for j in range(len(cls)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{res['model'].upper()} / {res['mode']} (acc={res['acc']:.3f})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def rq12(df, classes, seeds, epochs):
    results = []
    for seed in seeds:
        tr, va, te = make_splits(df, seed)
        print(f"\n=== seed {seed}: train={len(tr)} val={len(va)} test={len(te)} ===")
        for m in MODELS:
            for mode in MODES:
                print(f"  -> {m} / {mode}")
                r = run_experiment(tr, va, te, m, mode, classes, seed=seed, epochs=epochs)
                print(f"     TEST acc={r['acc']:.4f} macro-F1={r['macro_f1']:.4f}")
                results.append(r)
                if seed == seeds[0]:
                    plot_confusion(r, C.RESULTS / f"cm_{m}_{mode}.png")
    with open(C.RESULTS / "rq12_raw.json", "w") as f:
        json.dump(results, f)
    s = pd.DataFrame([{k: r[k] for k in ("model", "mode", "seed", "acc", "macro_f1")}
                      for r in results])
    summ = s.groupby(["model", "mode"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
    summ.to_csv(C.RESULTS / "rq12_summary.csv")
    print("\nChance level:", round(1 / len(classes), 3))
    print(summ)
    # accuracy-by-mode bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.35
    for i, m in enumerate(MODELS):
        means = [s[(s.model == m) & (s["mode"] == md)].acc.mean() for md in MODES]
        stds = [s[(s.model == m) & (s["mode"] == md)].acc.std() for md in MODES]
        ax.bar(np.arange(3) + i * w, means, w, yerr=np.nan_to_num(stds), label=m.upper())
    ax.axhline(1 / len(classes), ls="--", c="gray", label="chance")
    ax.set_xticks(np.arange(3) + w / 2); ax.set_xticklabels(["output only", "input only", "input+output"])
    ax.set_ylabel("Test accuracy"); ax.legend(); fig.tight_layout()
    fig.savefig(C.RESULTS / "rq2_accuracy_by_input.png", dpi=150); plt.close(fig)


def rq3(df, classes, epochs, min_n, seed=0):
    base = {}
    raw = C.RESULTS / "rq12_raw.json"
    if raw.exists():
        for r in json.load(open(raw)):
            if r["mode"] == "output" and r["seed"] == seed:
                base[r["model"]] = r["per_domain_acc"]
    rows = []
    for d, n in df["domain"].value_counts().items():
        if n < min_n:
            print(f"skip domain {d} (only {n} examples)"); continue
        train_all, test_d = df[df.domain != d], df[df.domain == d]
        tr, va = split_train_val(train_all, seed)
        for m in MODELS:
            print(f"\n[RQ3] held-out domain = {d}  model = {m}")
            r = run_experiment(tr, va, test_d, m, "output", classes, seed=seed, epochs=epochs)
            row = dict(held_out_domain=d, model=m, n_test=len(test_d),
                       ood_acc=r["acc"], ood_f1=r["macro_f1"],
                       in_domain_acc=base.get(m, {}).get(d, np.nan))
            row["drop"] = row["in_domain_acc"] - row["ood_acc"]
            for c, v in r["per_class_recall"].items():
                row[f"recall_{c}"] = v
            rows.append(row)
            print(f"   OOD acc={r['acc']:.3f}  in-domain acc={row['in_domain_acc']:.3f}")
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.RESULTS / "rq3_domain_shift.csv", index=False)
    print("\n", out.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", choices=["12", "3", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    a = ap.parse_args()
    C.RESULTS.mkdir(exist_ok=True)
    df, classes = load(a.quick)
    seeds = a.seeds if a.seeds is not None else ([0] if a.quick else C.SEEDS)
    epochs = 2 if a.quick else C.EPOCHS
    print("Classes:", classes, "| examples:", len(df))
    if a.rq in ("12", "all"):
        rq12(df, classes, seeds, epochs)
    if a.rq in ("3", "all"):
        rq3(df, classes, epochs, min_n=40 if a.quick else 150)
