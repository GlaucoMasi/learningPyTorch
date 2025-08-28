from pathlib import Path
SCRIPT_DIR = Path(__file__).parent

n_samples = 1000
epochs = 1000

import torch
# Make device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"