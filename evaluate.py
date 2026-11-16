"""Score predicted response matrices and probe pairs.

    score = 0.62 * ResponseScore + 0.38 * PairScore

ResponseScore = 0.70 * (all 18 levels exact) + 0.30 * weighted share of exact levels, where a true
level v weighs 1 + 0.4 v, so getting strong responses right counts more.
PairScore = 0.75 * (both probes exact) + 0.25 * (share of the two probes chosen correctly).
Both parts are averaged over boards. Range 0 to 1, higher is better.

Usage:
    python evaluate.py TRUE_CSV PRED_CSV
"""
import json
import sys

import numpy as np
import pandas as pd


def parse(df):
    matrix = np.array([json.loads(s) for s in df.response_matrix])
    pair = np.array([[int(s[0]), int(s[2])] for s in df.probe_pair])
    return matrix, pair


def score(true_matrix, pred_matrix, true_pair, pred_pair):
    tm = true_matrix.reshape(len(true_matrix), -1)
    pm = pred_matrix.reshape(len(pred_matrix), -1)
    hit = tm == pm
    weight = 1.0 + 0.4 * tm
    response = 0.70 * hit.all(1).mean() + 0.30 * ((weight * hit).sum(1) / weight.sum(1)).mean()
    overlap = (pred_pair[:, :, None] == true_pair[:, None, :]).any(2).sum(1) / 2.0
    plan = 0.75 * (pred_pair == true_pair).all(1).mean() + 0.25 * overlap.mean()
    return 0.62 * response + 0.38 * plan


if __name__ == "__main__":
    true = pd.read_csv(sys.argv[1])
    pred = pd.read_csv(sys.argv[2]).set_index("case_id").loc[true.case_id].reset_index()
    true_matrix, true_pair = parse(true)
    pred_matrix, pred_pair = parse(pred)
    print(f"score {score(true_matrix, pred_matrix, true_pair, pred_pair):.4f}")
