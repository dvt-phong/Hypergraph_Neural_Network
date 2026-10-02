# 15. Baseline CFIN (Feng et al., AAAI 2019, "Understanding Dropouts in MOOCs").
#
#   python src/15_cfin.py --mode both --seeds 1 --user-rule temporal
#
# Original code: github.com/wzfhaha/dropout_prediction (commit a99db2e), files
# feat_extract.py, preprocess.py, main.py, model.py. It is written for
# TensorFlow 1 (tf.contrib, placeholders), which no longer installs next to a
# current CUDA, so the model is ported here to PyTorch operation by operation:
#
#   activity a (115 fields)   embedding × value -> batch norm -> conv1d (width 5,
#                             stride 5, 512 filters) + ReLU: one vector per basic
#                             feature (23), each seen with its 4 smoothed versions
#   user u, course c fields   embedding × value -> flatten -> dense + ReLU = context (32)
#   attention                 softmax over the 23 vectors of
#                             w · ReLU(W [context ‖ vector] + b)
#   output                    [Σ attention · vector ‖ context] -> dropout -> dense 256
#                             -> batch norm -> ReLU -> dropout -> sigmoid
#   loss                      log loss + l2 1e-5 on every weight (tf.nn.l2_loss)
# Same initialization (normal 0.1 for embeddings, sqrt(2/(in+out)) otherwise),
# batch norm decay 0.995 and epsilon 1e-3, Adam lr 1e-4, batch 32 (the last
# partial batch is dropped, as in model.fit), 27 epochs, keep 0.9 / 0.9.
#
# Features (preprocess.py, main.feat_augment), with the protocol of 10_baseline_data.py:
#   23 basic activity counts: all events, "session" count (the original counts
#     rows with a session id, i.e. all events; 2_preprocess.py keeps no session
#     id), and 21 action counts (close_forum and close_info are not used)
#   context smoothing: mean and max of each count over the learner's enrollments
#     and over the course's enrollments. The original takes them over all
#     enrollments of train and test; here over the train enrollments allowed by
#     --user-rule (plus the target itself), and over the course's train enrollments.
#   user: age (2018 - birth, 0 outside 10-70), gender, user_enroll_num (the
#     enrollments counted for the learner above), cluster_label; course:
#     course_enroll_num, course_category.
#   Standardization (counts, age, enroll numbers) is fitted on train.
#   cluster_label: the original loads 5 user clusters from cluster/label_5_10time.npy,
#     computed outside the released code over all users. Here: k-means (k = 5)
#     fitted on the train enrollments' learner means of the 23 counts, applied to
#     every target's learner mean (so it follows --user-rule as well).
#   gender and category: the original maps "m"/"f" and looks categories up by the
#     numeric course key, which match nothing in user_info.csv / course_info.csv
#     (every value becomes 0); the intended codes are used here.
# The model is selected on validation AUPRC after every epoch (the original
# trains 27 epochs and reports the last one on test).

from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")
data_module = import_module("10_baseline_data")
log = data_module.log

VIDEO = ("seek_video", "play_video", "pause_video", "stop_video", "load_video")
PROBLEM = ("problem_get", "problem_check", "problem_save", "reset_problem", "problem_check_correct",
           "problem_check_incorrect")
FORUM = ("create_thread", "create_comment", "delete_thread", "delete_comment")
CLICK = ("click_info", "click_courseware", "click_about", "click_forum", "click_progress")
CLOSE = ("close_courseware",)
CFIN_ACTIONS = VIDEO + PROBLEM + FORUM + CLICK + CLOSE
EDUCATIONS = ("Bachelor's", "High", "Master's", "Primary", "Middle", "Associate", "Doctorate")
CATEGORIES = ("math", "physics", "electrical", "computer", "foreign language", "business", "economics",
              "biology", "medicine", "literature", "philosophy", "history", "social science", "art",
              "engineering", "education", "environment", "chemistry")
DEFAULT_SETTINGS = {
    "model": "cfin",
    "embedding_size": 32,
    "attn_size": 16,
    "conv_size": 512,
    "context_size": 32,
    "deep_size": 256,
    "dropout": 0.1,           # keep probability 0.9
    "epochs": 27,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "batch_norm_decay": 0.995,
    "l2_reg": 1e-5,
    "clusters": 5,
}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def basic_counts(split):
    counts = data_module.action_counts(split)
    columns = [config.ACTIONS.index(action) for action in CFIN_ACTIONS]
    total = counts.sum(axis=1, keepdims=True)
    return np.hstack([total, total, counts[:, columns]]).astype(np.float64)


# Mean and max of train rows `indices` per CSR row, plus the row's own vector.
def smoothed(train_values, indptr, indices, own):
    count = np.diff(indptr)
    gathered = train_values[indices]
    total = np.zeros_like(own)
    largest = own.copy()
    rows = np.flatnonzero(count)
    if len(rows):
        starts = indptr[:-1][rows]
        total[rows] = np.add.reduceat(gathered, starts, axis=0)
        largest[rows] = np.maximum(own[rows], np.maximum.reduceat(gathered, starts, axis=0))
    return (total + own) / (count + 1)[:, None], largest


def age_code(split):
    age = 2018 - split["birth"]
    return np.where(np.isfinite(age) & (age >= 10) & (age <= 70), age, 0.0)


def code(values, vocabulary):
    lookup = {value: index + 1 for index, value in enumerate(vocabulary)}
    return np.asarray([lookup.get(value, 0) for value in values], dtype=np.int64)


def build_features(cache, rule, clusters, seed):
    from sklearn.cluster import KMeans

    train = cache["train"]
    raw = {name: basic_counts(cache[name]) for name in config.SPLITS}
    count_scaler = data_module.fit_scaler(raw["train"])
    counts = {name: data_module.scale(values, count_scaler).astype(np.float64) for name, values in raw.items()}

    course_sizes = data_module.train_course_sizes(cache)
    course_sum = np.zeros((len(course_sizes), counts["train"].shape[1]))
    np.add.at(course_sum, train["course"], counts["train"])
    course_max = np.full_like(course_sum, -np.inf)
    np.maximum.at(course_max, train["course"], counts["train"])

    parts = {}
    for name in config.SPLITS:
        split, own = cache[name], counts[name]
        indptr, indices = data_module.same_user_train(cache, split, rule)
        user_mean, user_max = smoothed(counts["train"], indptr, indices, own)
        size = course_sizes[split["course"]].astype(np.float64)
        total = course_sum[split["course"]]
        if name == "train":  # the target is already one of its course's train rows
            course_mean, course_max_own = total / size[:, None], course_max[split["course"]]
        else:
            course_mean = (total + own) / (size + 1)[:, None]
            course_max_own = np.maximum(course_max[split["course"]], own)
            size = size + 1
        activity = np.stack([own, user_mean, user_max, course_mean, course_max_own], axis=2)
        parts[name] = {"activity": activity.reshape(len(own), -1), "user_mean": user_mean,
                       "user_enroll_num": np.diff(indptr) + 1.0, "course_enroll_num": size,
                       "age": age_code(split)}

    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(parts["train"]["user_mean"])
    numeric = ("age", "user_enroll_num", "course_enroll_num")
    scalers = {key: data_module.fit_scaler(parts["train"][key][:, None]) for key in numeric}
    gender_codes = {"male": 1, "female": 2}
    features = {}
    for name in config.SPLITS:
        split, part = cache[name], parts[name]
        scaled = {key: data_module.scale(part[key][:, None], scalers[key])[:, 0] for key in numeric}
        features[name] = {
            "activity": part["activity"].astype(np.float32),
            "user_numeric": np.stack([scaled["age"], scaled["user_enroll_num"]], axis=1),
            "user_categorical": np.stack([
                np.asarray([gender_codes.get(g, 0) for g in split["gender"]]),
                kmeans.predict(part["user_mean"]),
            ], axis=1),
            "course_numeric": scaled["course_enroll_num"][:, None],
            "course_categorical": code(split["category"], CATEGORIES)[:, None],
            "label": split["label"].astype(np.float32),
        }
    return features


# main.dataparse: every categorical field gets one embedding row per value, every
# numeric field one row; index and value per field (value 1 for categoricals).
def field_inputs(features, numeric_key, categorical_key):
    vocabularies = []
    offset = 0
    for column in range(features["train"][categorical_key].shape[1]):
        values = np.unique(np.concatenate([features[n][categorical_key][:, column] for n in config.SPLITS]))
        vocabularies.append((values, offset))
        offset += len(values)
    numeric_count = features["train"][numeric_key].shape[1]
    numeric_index = np.arange(offset, offset + numeric_count)
    inputs = {}
    for name in config.SPLITS:
        numeric = features[name][numeric_key]
        categorical = features[name][categorical_key]
        cat_index = np.stack([offset_ + np.searchsorted(values, categorical[:, c])
                              for c, (values, offset_) in enumerate(vocabularies)], axis=1)
        index = np.hstack([np.broadcast_to(numeric_index, numeric.shape), cat_index]).astype(np.int64)
        value = np.hstack([numeric, np.ones_like(categorical, dtype=np.float32)]).astype(np.float32)
        inputs[name] = (index, value)
    return offset + numeric_count, inputs


# ---------------------------------------------------------------------------
# Model (model.py, ported from TensorFlow 1)
# ---------------------------------------------------------------------------

def glorot(input_size, output_size, rng):
    scale = np.sqrt(2.0 / (input_size + output_size))
    weight = rng.normal(0, scale, size=(input_size, output_size))
    bias = rng.normal(0, scale, size=(1, output_size))
    return (nn.Parameter(torch.as_tensor(weight, dtype=torch.float32)),
            nn.Parameter(torch.as_tensor(bias, dtype=torch.float32)))


class CFIN(nn.Module):
    def __init__(self, a_field_size, u_feat_size, u_field_size, c_feat_size, c_field_size, settings, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        embedding, conv, context, attention, deep = (settings[k] for k in (
            "embedding_size", "conv_size", "context_size", "attn_size", "deep_size"))
        self.a_field_size, self.keep = a_field_size, 1.0 - settings["dropout"]
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
        momentum = 1.0 - settings["batch_norm_decay"]
        self.bn_conv = nn.BatchNorm1d(embedding, eps=1e-3, momentum=momentum)
        self.bn_0 = nn.BatchNorm1d(deep, eps=1e-3, momentum=momentum)

    # Weights of the original weights dict (tf.nn.l2_loss = Σ w² / 2); not batch norm.
    def l2_loss(self):
        return sum((p ** 2).sum() / 2 for n, p in self.named_parameters() if not n.startswith("bn_"))

    def forward(self, u_index, u_value, c_index, c_value, a_value):
        a = self.a_embeddings[None] * a_value[:, :, None]                     # B × 115 × E
        a = self.bn_conv(a.transpose(1, 2))                                   # B × E × 115
        a = F.relu(F.conv1d(a, self.conv_filter, self.conv_bias, stride=5))   # B × 512 × 23
        a = a.transpose(1, 2)                                                 # B × 23 × 512
        u = self.u_embeddings[u_index] * u_value[:, :, None]
        c = self.c_embeddings[c_index] * c_value[:, :, None]
        context = F.relu(torch.cat([u, c], dim=1).flatten(1) @ self.ctx_pool_weight + self.ctx_pool_bias)
        joint = torch.cat([context[:, None].expand(-1, a.shape[1], -1), a], dim=2)
        attention = F.relu(joint @ self.attn_out_1 + self.attn_bias_1) @ self.attn_out
        attention = torch.softmax(attention[:, :, 0], dim=1)
        deep = torch.cat([(attention[:, :, None] * a).sum(dim=1), context], dim=1)
        deep = F.dropout(deep, 1 - self.keep, self.training)
        deep = F.relu(self.bn_0(deep @ self.layer_0 + self.bias_0))
        deep = F.dropout(deep, 1 - self.keep, self.training)
        return torch.sigmoid(deep @ self.logistic_weight + self.logistic_bias)[:, 0]


# tf.losses.log_loss (epsilon 1e-7).
def log_loss(label, probability, epsilon=1e-7):
    return -(label * torch.log(probability + epsilon) + (1 - label) * torch.log(1 - probability + epsilon)).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def tensors(features, u_inputs, c_inputs, name, device):
    return [torch.as_tensor(array, device=device) for array in (
        *u_inputs[name], *c_inputs[name], features[name]["activity"], features[name]["label"])]


@torch.no_grad()
def predict(model, data, batch_size=4096):
    model.eval()
    probabilities = [model(*[t[start:start + batch_size] for t in data[:5]]).cpu()
                     for start in range(0, len(data[5]), batch_size)]
    return data[5].cpu().numpy().astype(int), torch.cat(probabilities).numpy()


def train_and_test(settings, seed, output_dir, device_name):
    data_module.set_seed(seed)
    device = data_module.resolve_device(device_name)
    run_name = data_module.make_run_name(settings, seed)
    log("setup", f"run={run_name}, device={device}, settings={settings}")
    cache = data_module.limit_cache(data_module.load_cache(output_dir), settings["limit"])
    features = build_features(cache, settings["user_rule"], settings["clusters"], seed)
    u_size, u_inputs = field_inputs(features, "user_numeric", "user_categorical")
    c_size, c_inputs = field_inputs(features, "course_numeric", "course_categorical")
    data = {name: tensors(features, u_inputs, c_inputs, name, device) for name in config.SPLITS}

    model = CFIN(features["train"]["activity"].shape[1], u_size, u_inputs["train"][0].shape[1],
                 c_size, c_inputs["train"][0].shape[1], settings, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], eps=1e-8)
    checkpoint_path = Path(config.RUNS) / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    history, best = [], {"score": -1.0, "epoch": 0}
    count, batch_size = len(data["train"][5]), settings["batch_size"]
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.as_tensor(rng.permutation(count), device=device)
        total = 0.0
        for batch in range(count // batch_size):
            rows = order[batch * batch_size:(batch + 1) * batch_size]
            *inputs, label = [t[rows] for t in data["train"]]
            loss = log_loss(label, model(*inputs)) + settings["l2_reg"] * model.l2_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        record = {"epoch": epoch, "loss": total / max(count // batch_size, 1)}
        log("train", f"epoch {epoch}: loss={record['loss']:.4f}")
        record["validation"] = data_module.epoch_validation(*predict(model, data["validation"]))
        if record["validation"]["auprc"] > best["score"]:
            best = {"score": record["validation"]["auprc"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "settings": settings, "seed": seed, "epoch": epoch},
                       checkpoint_path)
            log("checkpoint", f"new best val_auprc={best['score']:.4f} at epoch {epoch}")
        history.append(record)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    data_module.write_reports(run_name, settings, seed, history=history, best_epoch=best["epoch"],
                              checkpoint_path=checkpoint_path, validation=predict(model, data["validation"]),
                              test=predict(model, data["test"]))


if __name__ == "__main__":
    parser = data_module.base_parser("Train and test CFIN (Feng et al., 2019).", DEFAULT_SETTINGS)
    arguments = parser.parse_args()
    settings = data_module.settings_from(arguments, DEFAULT_SETTINGS)
    for seed in arguments.seeds:
        train_and_test(settings, seed, arguments.output_dir, arguments.device)
