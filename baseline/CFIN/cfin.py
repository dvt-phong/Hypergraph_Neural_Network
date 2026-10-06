# CFIN baseline (Feng et al., AAAI 2019, "Understanding Dropouts in MOOCs") on the split of the main model.
#
#   python baseline/CFIN/cfin.py --norm zscore --seeds 1 11 111 1111 11111
#   python baseline/CFIN/cfin.py --norm log1p  --seeds 1 11 111 1111 11111
# Needs data/processed/simple/{train,validation,test}.csv from src/2_preprocess.py. The first run
# caches one row per enrollment in baseline/CFIN/cache/; results go to baseline/CFIN/results.csv.
#
# Original code: baseline/CFIN/dropout_prediction (github.com/wzfhaha/dropout_prediction, a99db2e):
# feat_extract.py, preprocess.py, main.py, model.py. It needs TensorFlow 1, which does not install
# on Python 3.13, so the model is the PyTorch port of git tag full-hsl (src/15_cfin.py), op by op.
#
# Features, as the original:
#   23 counts per enrollment: all#count, session#count (the original counts log rows that have a
#     session id, i.e. all events), 21 action counts (close_forum is not used)
#   --norm zscore   x = (c − μ_train) / σ_train                    as CFIN (preprocess.py)
#   --norm log1p    x = (log(1 + c) − μ_train) / σ_train           the main model's earlier rule
#   feat_augment (main.py): mean and max of x over the learner's enrollments and over the course's
#     enrollments -> 23 × 5 = 115 activity fields
#   user: age = 2018 − birth (0 if missing or outside 10–70), user_enroll_num (z-score),
#         gender, cluster_label (embedding); course: course_enroll_num (z-score), category (embedding)
#   education is computed by the original but not used by main.py, so it is left out.
# Changes from the original, to follow the protocol of the main model:
#   - nodes and events: data/processed/simple/{train,validation,test}.csv (course days 0–34)
#   - μ, σ fitted on train only (the original fits on train + test)
#   - mean/max and enroll numbers over train + validation + test enrollments (the original:
#     train + test); features only, no labels
#   - the best of the 27 epochs by validation AUC is tested (the original reports epoch 27)
#   - gender "male"/"female" and category names are coded as intended: the original compares
#     with "m"/"f" and looks categories up by the numeric course id, so both are 0 for everyone
#   - cluster_label: the original loads 5 precomputed user clusters (cluster/label_5_10time.npy,
#     how they were made is not released); here k-means (k = 5) on each learner's mean x.

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "processed" / "simple"
CACHE = HERE / "cache"
RESULTS_CSV = HERE / "results.csv"
SPLITS = ("train", "validation", "test")
THRESHOLD = 0.5

# The 22 actions of the logs (src/0_config.py) and the 21 of CFIN (feat_extract.py).
ACTIONS = ("seek_video", "play_video", "pause_video", "stop_video", "load_video",
           "problem_get", "problem_check", "problem_save", "reset_problem", "problem_check_correct",
           "problem_check_incorrect", "create_thread", "create_comment", "delete_thread", "delete_comment",
           "close_forum", "click_info", "click_courseware", "click_about", "click_forum", "click_progress",
           "close_courseware")
CFIN_ACTIONS = ("seek_video", "play_video", "pause_video", "stop_video", "load_video",
                "problem_get", "problem_check", "problem_save", "reset_problem", "problem_check_correct",
                "problem_check_incorrect", "create_thread", "create_comment", "delete_thread",
                "delete_comment", "click_info", "click_courseware", "click_about", "click_forum",
                "click_progress", "close_courseware")
CATEGORIES = ("math", "physics", "electrical", "computer", "foreign language", "business", "economics",
              "biology", "medicine", "literature", "philosophy", "history", "social science", "art",
              "engineering", "education", "environment", "chemistry")   # preprocess.py, code = index + 1
GENDERS = {"male": 1, "female": 2}                                       # intended "m" -> 1, "f" -> 2

SETTINGS = {                       # main.py params
    "embedding_size": 32,
    "attn_size": 16,
    "conv_size": 512,
    "context_size": 32,
    "deep_size": 256,
    "dropout": 0.1,                # keep probability 0.9
    "epochs": 27,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "batch_norm_decay": 0.995,
    "l2_reg": 1e-5,
    "clusters": 5,
}
RESULT_COLUMNS = (
    "time", "model", "norm", "seed", "epochs", "best_epoch",
    "val_auc", "val_auprc", "val_f1",
    "test_auc", "test_auprc", "test_accuracy", "test_precision", "test_recall", "test_f1",
    "minutes",
)


def log(scope, message):
    print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Data: one row per enrollment (node) with its 22 action counts, read once and cached
# ---------------------------------------------------------------------------

# Input:  split name.
# Output: dict of per-node arrays (node i at row i): user, course, label, gender, birth,
#         category, counts [N, 22] (counts[v, a] = events of v with action a in days 0–34).
def read_split(name):
    cache_path = CACHE / f"{name}.npz"
    if cache_path.exists():
        return dict(np.load(cache_path))
    log("data", f"reading {DATA / f'{name}.csv'} (once, then cached)")
    action_index = {action: index for index, action in enumerate(ACTIONS)}
    meta_parts, count_parts = [], []
    columns = ["node_id", "user_id", "course_id", "label", "gender", "birth", "category", "action"]
    for chunk in pd.read_csv(DATA / f"{name}.csv", usecols=columns, chunksize=4_000_000,
                             dtype={"gender": str, "category": str, "action": str, "course_id": str}):
        meta_parts.append(chunk.drop_duplicates("node_id").drop(columns="action"))
        events = chunk[chunk["action"].notna()]                         # no-event nodes have action ""
        count_parts.append(events.groupby(["node_id", "action"]).size().reset_index())
    meta = pd.concat(meta_parts).drop_duplicates("node_id").sort_values("node_id")
    node_count = len(meta)
    if not np.array_equal(meta["node_id"].to_numpy(), np.arange(node_count)):
        raise ValueError(f"{name}: node ids are not 0..N-1")
    pairs = pd.concat(count_parts).groupby(["node_id", "action"])[0].sum().reset_index()
    counts = np.zeros((node_count, len(ACTIONS)), dtype=np.int64)
    counts[pairs["node_id"].to_numpy(), pairs["action"].map(action_index).to_numpy()] = pairs[0].to_numpy()
    data = {
        "user": meta["user_id"].to_numpy(np.int64),
        "course": meta["course_id"].to_numpy(str),
        "label": meta["label"].to_numpy(np.int64),
        "gender": meta["gender"].fillna("").to_numpy(str),
        "birth": meta["birth"].to_numpy(np.float64),
        "category": meta["category"].fillna("").to_numpy(str),
        "counts": counts,
    }
    CACHE.mkdir(exist_ok=True)
    np.savez(cache_path, **data)
    log("data", f"{name}: {node_count} nodes, {counts.sum()} events, cached to {cache_path}")
    return data


# Fit μ, σ per column on train rows; σ = 0 -> 1.
def fit_scaler(train_values):
    mean = train_values.mean(axis=0)                                     # μ_train
    std = train_values.std(axis=0)                                       # σ_train (ddof 0, as StandardScaler)
    std[std == 0] = 1.0
    return mean, std


# ---------------------------------------------------------------------------
# Features (feat_extract.py, preprocess.py, main.feat_augment)
# ---------------------------------------------------------------------------

def build_features(data, norm, seed):
    sizes = [len(data[name]["label"]) for name in SPLITS]
    cfin_columns = [ACTIONS.index(action) for action in CFIN_ACTIONS]
    raw = {}
    for name in SPLITS:
        counts = data[name]["counts"].astype(np.float64)
        total = counts.sum(axis=1, keepdims=True)                        # all#count = Σ_a c_v[a]
        raw[name] = np.hstack([total, total, counts[:, cfin_columns]])   # [all#count, session#count, 21 #num]
        if norm == "log1p":
            raw[name] = np.log1p(raw[name])                              # c -> log(1 + c)
    mean, std = fit_scaler(raw["train"])
    x = np.vstack([(raw[name] - mean) / std for name in SPLITS])         # x = (c − μ_train) / σ_train

    users = np.concatenate([data[name]["user"] for name in SPLITS])
    courses = np.concatenate([data[name]["course"] for name in SPLITS])
    frame = pd.DataFrame(x)
    by_user, by_course = frame.groupby(users), frame.groupby(courses)
    user_mean = by_user.transform("mean").to_numpy()                     # mean of x over the learner's enrollments
    user_max = by_user.transform("max").to_numpy()
    course_mean = by_course.transform("mean").to_numpy()                 # mean of x over the course's enrollments
    course_max = by_course.transform("max").to_numpy()
    # Field order of main.feat_augment: f, f#user#mean, f#user#max, f#course#mean, f#course#max.
    activity = np.stack([x, user_mean, user_max, course_mean, course_max], axis=2).reshape(len(x), -1)

    user_enroll_num = by_user[0].transform("size").to_numpy(np.float64)      # enrollments of the learner
    course_enroll_num = by_course[0].transform("size").to_numpy(np.float64)  # enrollments of the course
    birth = np.concatenate([data[name]["birth"] for name in SPLITS])
    age = 2018 - birth                                                   # a = 2018 − birth (preprocess.py)
    age = np.where(np.isfinite(age) & (age >= 10) & (age <= 70), age, 0.0)   # missing or outside 10–70 -> 0

    learner_means = by_user.mean()                                       # one row per learner
    kmeans = KMeans(n_clusters=SETTINGS["clusters"], n_init=10, random_state=seed).fit(learner_means.to_numpy())
    cluster = pd.Series(kmeans.labels_, index=learner_means.index).loc[users].to_numpy()

    gender = np.array([GENDERS.get(g, 0) for name in SPLITS for g in data[name]["gender"]])
    category_code = {category: index + 1 for index, category in enumerate(CATEGORIES)}
    category = np.array([category_code.get(c, 0) for name in SPLITS for c in data[name]["category"]])

    numeric = np.stack([age, user_enroll_num, course_enroll_num], axis=1)
    numeric_mean, numeric_std = fit_scaler(numeric[:sizes[0]])
    numeric = (numeric - numeric_mean) / numeric_std                     # z-score with train μ, σ

    features = {}
    offsets = np.cumsum([0] + sizes)
    for index, name in enumerate(SPLITS):
        rows = slice(offsets[index], offsets[index + 1])
        features[name] = {
            "activity": activity[rows].astype(np.float32),
            "user_numeric": numeric[rows, :2].astype(np.float32),        # age, user_enroll_num
            "user_categorical": np.stack([gender[rows], cluster[rows]], axis=1),
            "course_numeric": numeric[rows, 2:].astype(np.float32),      # course_enroll_num
            "course_categorical": category[rows][:, None],
            "label": data[name]["label"].astype(np.float32),
        }
    return features


# main.dataparse: each categorical field gets one embedding row per value, each numeric field one
# row; per field an index and a value (value 1 for categoricals).
def field_inputs(features, numeric_key, categorical_key):
    vocabularies, offset = [], 0
    for column in range(features["train"][categorical_key].shape[1]):
        values = np.unique(np.concatenate([features[n][categorical_key][:, column] for n in SPLITS]))
        vocabularies.append((values, offset))
        offset += len(values)
    numeric_count = features["train"][numeric_key].shape[1]
    numeric_index = np.arange(offset, offset + numeric_count)
    inputs = {}
    for name in SPLITS:
        numeric = features[name][numeric_key]
        categorical = features[name][categorical_key]
        categorical_index = np.stack([start + np.searchsorted(values, categorical[:, c])
                                      for c, (values, start) in enumerate(vocabularies)], axis=1)
        index = np.hstack([np.broadcast_to(numeric_index, numeric.shape), categorical_index]).astype(np.int64)
        value = np.hstack([numeric, np.ones(categorical.shape, dtype=np.float32)]).astype(np.float32)
        inputs[name] = (index, value)
    return offset + numeric_count, inputs


# ---------------------------------------------------------------------------
# Model (model.py, ported from TensorFlow 1)
#   activity (115 fields)  embedding × value -> batch norm -> conv1d (width 5, stride 5, 512) + ReLU:
#                          one vector per basic count (23), seen with its 4 smoothed versions
#   user, course fields    embedding × value -> flatten -> dense + ReLU = context (32)
#   attention              softmax over the 23 vectors of w · ReLU(W [context ‖ vector] + b)
#   output                 [Σ attention · vector ‖ context] -> dropout -> dense 256 -> batch norm
#                          -> ReLU -> dropout -> sigmoid
# ---------------------------------------------------------------------------

def glorot(input_size, output_size, rng):
    scale = np.sqrt(2.0 / (input_size + output_size))
    weight = rng.normal(0, scale, size=(input_size, output_size))
    bias = rng.normal(0, scale, size=(1, output_size))
    return (nn.Parameter(torch.as_tensor(weight, dtype=torch.float32)),
            nn.Parameter(torch.as_tensor(bias, dtype=torch.float32)))


class CFIN(nn.Module):
    def __init__(self, a_field_size, u_feat_size, u_field_size, c_feat_size, c_field_size, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        embedding, conv, context, attention, deep = (SETTINGS[k] for k in (
            "embedding_size", "conv_size", "context_size", "attn_size", "deep_size"))
        self.a_embeddings = nn.Parameter(torch.randn(a_field_size, embedding) * 0.1)
        self.u_embeddings = nn.Parameter(torch.randn(u_feat_size, embedding) * 0.1)
        self.c_embeddings = nn.Parameter(torch.randn(c_feat_size, embedding) * 0.1)
        self.ctx_pool_weight, self.ctx_pool_bias = glorot((u_field_size + c_field_size) * embedding, context, rng)
        conv_weight, conv_bias = glorot(5 * embedding, conv, rng)
        # TF filter [width 5, in, out] -> torch [out, in, width]
        self.conv_filter = nn.Parameter(conv_weight.data.reshape(5, embedding, conv).permute(2, 1, 0).contiguous())
        self.conv_bias = nn.Parameter(conv_bias.data.reshape(conv))
        self.attn_out_1, self.attn_bias_1 = glorot(conv + context, attention, rng)
        self.attn_out, _ = glorot(attention, 1, rng)
        self.layer_0, self.bias_0 = glorot(conv + context, deep, rng)
        self.logistic_weight, _ = glorot(deep, 1, rng)
        self.logistic_bias = nn.Parameter(torch.tensor(0.01))
        momentum = 1.0 - SETTINGS["batch_norm_decay"]
        self.bn_conv = nn.BatchNorm1d(embedding, eps=1e-3, momentum=momentum)
        self.bn_0 = nn.BatchNorm1d(deep, eps=1e-3, momentum=momentum)

    # tf.nn.l2_loss = Σ w² / 2 over the weights dict (not batch norm).
    def l2_loss(self):
        return sum((p ** 2).sum() / 2 for n, p in self.named_parameters() if not n.startswith("bn_"))

    def forward(self, u_index, u_value, c_index, c_value, a_value):
        dropout = SETTINGS["dropout"]
        a = self.a_embeddings[None] * a_value[:, :, None]                     # B × 115 × E
        a = self.bn_conv(a.transpose(1, 2))                                   # B × E × 115
        a = F.relu(F.conv1d(a, self.conv_filter, self.conv_bias, stride=5))   # B × 512 × 23
        a = a.transpose(1, 2)                                                 # B × 23 × 512
        u = self.u_embeddings[u_index] * u_value[:, :, None]
        c = self.c_embeddings[c_index] * c_value[:, :, None]
        context = F.relu(torch.cat([u, c], dim=1).flatten(1) @ self.ctx_pool_weight + self.ctx_pool_bias)
        joint = torch.cat([context[:, None].expand(-1, a.shape[1], -1), a], dim=2)
        attention = F.relu(joint @ self.attn_out_1 + self.attn_bias_1) @ self.attn_out
        attention = torch.softmax(attention[:, :, 0], dim=1)                  # α_k over the 23 vectors
        deep = torch.cat([(attention[:, :, None] * a).sum(dim=1), context], dim=1)   # [Σ_k α_k·a_k ‖ context]
        deep = F.dropout(deep, dropout, self.training)
        deep = F.relu(self.bn_0(deep @ self.layer_0 + self.bias_0))
        deep = F.dropout(deep, dropout, self.training)
        return torch.sigmoid(deep @ self.logistic_weight + self.logistic_bias)[:, 0]


# tf.losses.log_loss (epsilon 1e-7).
def log_loss(label, probability, epsilon=1e-7):
    return -(label * torch.log(probability + epsilon) + (1 - label) * torch.log(1 - probability + epsilon)).mean()


# ---------------------------------------------------------------------------
# Training and evaluation (same metrics as src/9_train.py)
# ---------------------------------------------------------------------------

def metrics(labels, probabilities):
    predicted = probabilities >= THRESHOLD                              # ŷ = 1[p ≥ 0.5]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=[1], zero_division=0)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(np.mean(predicted == labels)),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


@torch.no_grad()
def predict(model, data, batch_size=4096):
    model.eval()
    probabilities = [model(*[t[start:start + batch_size] for t in data[:5]])
                     for start in range(0, len(data[5]), batch_size)]
    return data[5].cpu().numpy().astype(int), torch.cat(probabilities).cpu().numpy()


def run(norm, seed, raw_data, limit, device):
    started = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    features = build_features(raw_data, norm, seed)
    u_size, u_inputs = field_inputs(features, "user_numeric", "user_categorical")
    c_size, c_inputs = field_inputs(features, "course_numeric", "course_categorical")
    data = {name: [torch.as_tensor(np.ascontiguousarray(a), device=device) for a in (
        *u_inputs[name], *c_inputs[name], features[name]["activity"], features[name]["label"])]
        for name in SPLITS}
    if limit:
        data = {name: [t[:limit] for t in tensors] for name, tensors in data.items()}

    model = CFIN(features["train"]["activity"].shape[1], u_size, u_inputs["train"][0].shape[1],
                 c_size, c_inputs["train"][0].shape[1], seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"], eps=1e-8)
    rng = np.random.default_rng(seed)
    count, batch_size = len(data["train"][5]), SETTINGS["batch_size"]
    best = {"auc": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, SETTINGS["epochs"] + 1):
        model.train()
        order = torch.as_tensor(rng.permutation(count), device=device)
        total = 0.0
        for batch in range(count // batch_size):                          # last partial batch dropped
            rows = order[batch * batch_size:(batch + 1) * batch_size]
            *inputs, label = [t[rows] for t in data["train"]]
            loss = log_loss(label, model(*inputs)) + SETTINGS["l2_reg"] * model.l2_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        validation_auc = roc_auc_score(*predict(model, data["validation"]))
        if validation_auc > best["auc"]:                                  # best = argmax_epoch AUC_val
            best = {"auc": validation_auc, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        log("train", f"{norm} seed {seed} epoch {epoch}: loss={total / max(count // batch_size, 1):.4f}, "
                     f"val_auc={validation_auc:.4f} (best {best['auc']:.4f} at epoch {best['epoch']})")

    model.load_state_dict(best["state"])
    scores = {name: metrics(*predict(model, data[name])) for name in ("validation", "test")}
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": "CFIN", "norm": norm, "seed": seed,
        "epochs": SETTINGS["epochs"], "best_epoch": best["epoch"],
        **{f"val_{k}": round(scores["validation"][k], 6) for k in ("auc", "auprc", "f1")},
        **{f"test_{k}": round(v, 6) for k, v in scores["test"].items()},
        "minutes": round((time.time() - started) / 60, 2),
    }
    log("result", f"{norm} seed {seed}: test_auc={row['test_auc']:.4f}, val_auc={row['val_auc']:.4f}")
    if limit:
        return
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test CFIN (Feng et al., 2019).")
    parser.add_argument("--norm", choices=("zscore", "log1p"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 11, 111, 1111, 11111])
    parser.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N nodes per split, no result row")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto (cuda when available), cpu or cuda")
    arguments = parser.parse_args()
    SETTINGS["epochs"] = arguments.epochs
    if arguments.threads:
        torch.set_num_threads(arguments.threads)
    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    log("setup", f"norm={arguments.norm}, seeds={arguments.seeds}, device={device}")
    raw_data = {name: read_split(name) for name in SPLITS}
    for seed in arguments.seeds:
        run(arguments.norm, seed, raw_data, arguments.limit, device)
