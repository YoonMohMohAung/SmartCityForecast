import numpy as np
from src.models.model_utils import evaluate


def run_baseline(train, test):
    # naive: last value
    preds = np.repeat(train.iloc[-1], len(test))
    evaluate("Baseline (Naive)", test.values, preds)
