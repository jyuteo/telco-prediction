import os
import random
import numpy as np
from joblib import load, dump


def reset_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def load_model(path):
    return load(path)


def save_model(model, path):
    dump(model, path)