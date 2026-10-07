# Who Wrote It? Identifying LLMs from Their Responses (CMPSC 448)

CNN and BiLSTM classifiers that predict which LLM family wrote a response.

## Data
Public dataset **UltraFeedback** (`openbmb/UltraFeedback`, Hugging Face). Each row has a
prompt and completions from many LLMs labelled with the model name. `build_dataset.py`
groups models into families (`config.FAMILIES`) and samples a class-balanced dataset
`(llm_name, input, output)` plus a `domain` column (the prompt's source task type).
Splits are grouped by prompt so a prompt never appears in both train and test.

## Setup (VS Code terminal)
```
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run order
```
python build_dataset.py                 # -> data/dataset.csv
python run_experiments.py --quick       # fast smoke test of the whole pipeline
python run_experiments.py --rq 12       # RQ1 + RQ2  (CNN/LSTM x output/input/both, 3 seeds)
python run_experiments.py --rq 3        # RQ3 leave-one-domain-out
python analysis_rq4.py --nn             # RQ4 feature analysis + signal-removal ablation
```
All tables/figures are written to `results/`.

## Files
| file | purpose |
|---|---|
| config.py | all hyperparameters and settings |
| build_dataset.py | data collection / balancing |
| data_utils.py | tokenizer, vocab, encoding, prompt-grouped splits |
| models.py | TextCNN and BiLSTM |
| train.py | training loop (Adam, early stopping on val macro-F1) and evaluation |
| run_experiments.py | RQ1, RQ2, RQ3 |
| analysis_rq4.py | RQ4 |
