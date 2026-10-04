# Paired DeLong test of test AUC between two scripts/run_all.sh runs.
#
#   python scripts/delong.py result/<model A> result/<model B> [result/<model C> ...]
#
# The first folder is compared with each of the others, seed by seed: for
# every seed found in both folders, reads reports/*_seed_<seed>_test_probs.npz
# (written by 6_train.py), checks that both hold the same
# test targets in the same order, and prints AUC A, AUC B, their difference,
# z and the two-sided p-value. A seed-free model (logreg) is matched to every
# seed of the other folder.
#
# DeLong, DeLong & Clarke-Pearson (Biometrics, 1988); fast version of Sun & Xu
# (IEEE Signal Processing Letters, 2014). The test treats predictions as fixed,
# so it does not cover variation between training seeds: report the seed mean
# and std next to it.

import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

SEED_PATTERN = re.compile(r"_seed_(\d+)_test_probs\.npz$")


def midrank(values):
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        stop = start
        while stop < len(values) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1) + 1
        start = stop
    result = np.empty(len(values))
    result[order] = ranks
    return result


# AUC of a and b on the same labels, z and two-sided p of their difference.
def delong(labels, a, b):
    positive, negative = labels == 1, labels == 0
    m, n = positive.sum(), negative.sum()
    aucs, v01, v10 = [], [], []
    for scores in (a, b):
        tx, ty = midrank(scores[positive]), midrank(scores[negative])
        tz = midrank(np.concatenate([scores[positive], scores[negative]]))
        aucs.append(tz[:m].sum() / m / n - (m + 1) / (2 * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    covariance = np.cov(np.array(v01)) / m + np.cov(np.array(v10)) / n
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    return aucs[0], aucs[1], z, 2 * norm.sf(abs(z))


def probability_files(run_dir):
    files = {}
    for path in (Path(run_dir) / "reports").glob("*_test_probs.npz"):
        match = SEED_PATTERN.search(path.name)
        if match:
            files[int(match.group(1))] = path
    if not files:
        sys.exit(f"No *_seed_<n>_test_probs.npz in {Path(run_dir) / 'reports'}")
    return files


def load(path):
    with np.load(path) as data:
        return data["labels"].astype(int), data["probabilities"]


def main(first, others):
    a_files = probability_files(first)
    for other in others:
        b_files = probability_files(other)
        print(f"== {Path(first).name}  vs  {Path(other).name}")
        for seed in sorted(a_files):
            b_path = b_files.get(seed) or (next(iter(b_files.values())) if len(b_files) == 1 else None)
            if b_path is None:
                continue
            labels, a = load(a_files[seed])
            b_labels, b = load(b_path)
            if not np.array_equal(labels, b_labels):
                sys.exit(f"Test targets differ between {a_files[seed].name} and {b_path.name}")
            auc_a, auc_b, z, p = delong(labels, a, b)
            print(f"  seed {seed:>5}: AUC {auc_a:.4f} vs {auc_b:.4f}  Δ={auc_a - auc_b:+.4f}  "
                  f"z={z:6.2f}  p={p:.2e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python scripts/delong.py result/<model A> result/<model B> [...]")
    main(sys.argv[1], sys.argv[2:])
