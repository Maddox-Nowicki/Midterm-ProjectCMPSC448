"""Step 1: build data/dataset.csv with columns
   id, prompt_id, llm_name, model, domain, input, output

Source: the public UltraFeedback dataset (openbmb/UltraFeedback on Hugging Face).
Each prompt has completions written by several different LLMs, and each
completion is labelled with the model that wrote it. We group specific models
into families (see config.FAMILIES) and sample a balanced dataset.
"""
import argparse
import pandas as pd
import config as C


def family_of(model_name):
    m = str(model_name).lower()
    for fam, keys in C.FAMILIES.items():
        if any(k in m for k in keys):
            return fam
    return None


def domain_of(source):
    s = str(source).lower()
    return "flan" if s.startswith("flan") else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=C.N_PER_CLASS)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    print("Columns:", ds.column_names)

    rows = []
    for pid, ex in enumerate(ds):
        for c in ex["completions"]:
            fam = family_of(c["model"])
            resp = c.get("response")
            if fam is None or not isinstance(resp, str) or len(resp.strip()) < 20:
                continue
            rows.append(dict(prompt_id=pid, llm_name=fam, model=c["model"],
                             domain=domain_of(ex["source"]),
                             input=ex["instruction"][:2000],
                             output=resp.strip()[:4000]))
    df = pd.DataFrame(rows)
    print("\nAvailable examples per family:\n", df.llm_name.value_counts())
    print("\nSpecific models per family:\n", df.groupby(["llm_name", "model"]).size())

    n = min(args.n_per_class, df.llm_name.value_counts().min())
    df = pd.concat([g.sample(n, random_state=C.SEED) for _, g in df.groupby("llm_name")])
    df = df.sample(frac=1, random_state=C.SEED).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    C.DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(C.DATA_PATH, index=False)
    print(f"\nSaved {len(df)} rows ({n} per class) to {C.DATA_PATH}")
    print("\nFamily x domain counts:\n", pd.crosstab(df.llm_name, df.domain))


if __name__ == "__main__":
    main()
