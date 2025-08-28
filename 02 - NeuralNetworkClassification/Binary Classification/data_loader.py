import torch
from sklearn.datasets import make_circles
import config as cfg
from sklearn.model_selection import train_test_split

def load_circles():
    X, y = make_circles(cfg.n_samples, noise=0.03, random_state=42)

    import matplotlib.pyplot as plt
    plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.savefig(cfg.SCRIPT_DIR / "plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_line():
    weight = 0.7
    bias = 0.3
    start = 0
    end = 1
    step = 0.01

    X = torch.arange(start, end, step).unsqueeze(dim = 1)
    y = weight * X + bias

    train_split = int(0.8 * len(X))

    return X[:train_split], X[train_split:], y[:train_split], y[train_split:]