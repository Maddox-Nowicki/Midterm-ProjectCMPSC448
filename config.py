"""Central configuration. Edit values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "dataset.csv"
RESULTS = ROOT / "results"

# ---- data ----------------------------------------------------------------
# class name -> substrings that identify a model name belonging to that family
FAMILIES = {
    "GPT": ["gpt-4", "gpt-3.5"],
    "Llama-2": ["llama-2"],
    "Bard": ["bard"],
    "Falcon": ["falcon"],
}
N_PER_CLASS = 2500          # balanced: same number of examples per LLM family
SEED = 42

# ---- text / vocab --------------------------------------------------------
MAX_IN = 64                 # max tokens kept from the prompt
MAX_OUT = 256               # max tokens kept from the response
MIN_FREQ = 2
MAX_VOCAB = 30000

# ---- training ------------------------------------------------------------
SEEDS = [0, 1, 2]           # repeated runs -> mean +- std
BATCH = 64
EPOCHS = 8
PATIENCE = 2                # early stopping on validation macro-F1
LR = 1e-3
EMB_DIM = 128
