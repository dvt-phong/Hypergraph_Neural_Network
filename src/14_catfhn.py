# 14. Baseline CA-TFHN (Liang et al., ICONIP 2023): Classmates Augmented Time-Flow
#     Hybrid Network.
#
#   .venvs/catfhn/bin/python src/14_catfhn.py --mode both --seeds 1 --user-rule temporal
#
# Original code: baseline/CA-TFHN (github.com/codeds27/CA-TFHN, commit f8ef659).
# The classifier, LGB of src/models.py (context embedding, 2-layer GraphSAGE on the
# classmates graph, TFHN = LSTM + self-attention twice over 5 weeks, weighted sum,
# DNN), is imported unchanged. Its pipeline, re-run here on the common split:
#
#   1. features (dataprocess/0-1): counts of 22 actions on each of the 35 days;
#      age, gender, education, cluster label, number of the learner's enrollments;
#      course enrollments, course category
#   2. link prediction (linkprediction/linkprediction_models.py): a learner-course
#      bipartite graph, GraphSAGE encoder (learned id embedding ‖ linear of the
#      features), dot-product decoder trained to recover enrollments (10 epochs,
#      best epoch by link AUC); predicted links: sigmoid ≥ 0.6, plus the real ones
#   3. strong classmates graph (graphgeneration/0-1): two enrollments of the same
#      course are linked when their learners' rows of the link matrix have cosine
#      ≥ 0.95; node context = [learner features ‖ mean course features] (7) and
#      enhanced context = [learner embedding ‖ mean course embedding] (32)
#   4. train.py: LGB with neighbor sampling [8, 4], batch 256, lr 1e-4, 15 epochs
#
# What differs from the original, and why:
#   - models.LGB.forward starts the 5-week sequence with `if i == 1`, so at i = 0
#     it concatenates None and stops with a TypeError; FixedLGB below makes it
#     `if i == 0` (week 1 first). Nothing else in models.py changes.
#   - Learner nodes follow --user-rule (10_baseline_data.py). Rule "any": one node
#     per learner, linked to all their courses, as the original (but with train
#     enrollments only, plus the target's own course). Rule "temporal": a learner
#     node per point in time, "the learner as of the target's course start",
#     linked to the courses that started no later; a course only receives from the
#     learner nodes of its own start, so no embedding mixes in a later course.
#     Learner features (enrollment count, cluster) use the same courses.
#   - Classmates graph: train enrollments send to every enrollment of their course
#     (cosine ≥ 0.95); validation/test enrollments only receive, so targets never
#     see each other (the original links all enrollments, test included). Courses
#     with thousands of near-identical learners would give millions of edges, while
#     the sampler draws only 8 then 4; each node keeps a random 64 of its classmates.
#   - Link prediction runs on the whole learner-course graph instead of the
#     sampled LinkNeighborLoader [3, 90]; edge split as RandomLinkSplit (10%
#     validation, 30% of the rest supervision-only, 3 negatives per positive).
#     The [8, 4] neighbor sampling of train.py is done here without pyg-lib.
#   - cluster_label: the original uses CFIN's cluster/label_5_10time.npy, computed
#     outside the released code; here k-means (k = 5) as in 15_cfin.py.
#     Gender: the original maps "m"/"f", which never occur (all 0); male 1, female 2 here.
#   - Scalers and course statistics on train; best of 15 epochs on validation AUPRC.

from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

config = import_module("0_config")
data_module = import_module("10_baseline_data")
cfin_module = import_module("15_cfin")
log = data_module.log

# 0_process_user_activity_logs.py: video 5, problem 6, forum 5, click 5, close 1
CATFHN_ACTIONS = (cfin_module.VIDEO + cfin_module.PROBLEM + cfin_module.FORUM + ("close_forum",)
                  + cfin_module.CLICK + cfin_module.CLOSE)
CATEGORIES = cfin_module.CATEGORIES
DEFAULT_SETTINGS = {
    "model": "catfhn",
    "learning_rate": 1e-4,
    "batch_size": 256,
    "epochs": 15,
    "link_epochs": 10,
    "link_learning_rate": 1e-3,
    "link_batch_size": 256,
    "link_threshold": 0.6,
    "similarity": 0.95,
    "max_classmates": 64,
    "clusters": 5,
}
# train.py
PARAMETERS = {
    "activity_num": 22, "sta_day": 35, "week_count": 5, "select_count": 5,
    "org_context_feat_len": 7, "enhanced_context_feat_len": 32, "context_each_embed": 16, "context_all_len": 16,
    "input_features": 16, "hidden_features": 32, "output_features": 16,
    "lstm_input_features": 184, "lstm_hidden_features": 128, "lstm_hidden_num_layers": 1,
    "num_attention_heads": 1, "attention_features": 64,
    "l2_input_features": 64, "l2_hidden_features": 32, "l2_hidden_num_layers": 1,
    "s2_num_attention_heads": 1, "s2_attention_features": 16,
    "ws_num_attention_heads": 1, "ws_input_features": 32, "ws_attention_features": 16,
    "dnn_input_f1": 16, "dnn_hidden_f1": 16, "dnn_hidden_f2": 8, "dnn_hidden_f3": 4, "dnn_output": 1,
}


# ---------------------------------------------------------------------------
# 1. Enrollment features and learner nodes
# ---------------------------------------------------------------------------

class Enrollments:
    """All enrollments of the three splits, in one index: train, validation, test."""

    def __init__(self, cache, rule, clusters, seed):
        from sklearn.cluster import KMeans

        self.offsets = np.cumsum([0] + [len(cache[n]["label"]) for n in config.SPLITS])
        self.count = int(self.offsets[-1])
        joined = {key: np.concatenate([cache[n][key] for n in config.SPLITS])
                  for key in ("user_id", "course", "label", "start", "birth", "gender", "education", "category")}
        self.course, self.label, self.start = joined["course"], joined["label"], joined["start"]
        train = cache["train"]

        # Courses visible to each enrollment: its own + the learner's train
        # enrollments allowed by the rule; and the learner mean of CFIN's counts.
        counts = {n: cfin_module.basic_counts(cache[n]) for n in config.SPLITS}
        scaler = data_module.fit_scaler(counts["train"])
        scaled = {n: data_module.scale(v, scaler).astype(np.float64) for n, v in counts.items()}
        visible, user_mean = [], []
        for name in config.SPLITS:
            indptr, indices = data_module.same_user_train(cache, cache[name], rule)
            user_mean.append(cfin_module.smoothed(scaled["train"], indptr, indices, scaled[name])[0])
            for target in range(len(cache[name]["label"])):
                courses = train["course"][indices[indptr[target]:indptr[target + 1]]]
                visible.append(tuple(sorted(set(courses.tolist()) | {int(cache[name]["course"][target])})))
        user_mean = np.concatenate(user_mean)
        is_train = np.arange(self.count) < self.offsets[1]
        cluster = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(user_mean[is_train]).predict(user_mean)

        # Learner nodes: one per (learner, visible courses).
        keys = {}
        self.learner = np.asarray([keys.setdefault((u, v), len(keys))
                                   for u, v in zip(joined["user_id"].tolist(), visible)])
        first = np.full(len(keys), -1)
        first[self.learner[::-1]] = np.arange(self.count)[::-1]
        self.learner_courses = [visible[i] for i in first]
        self.learner_time = self.start[first]  # its own course is the latest one

        age = 2023 - joined["birth"]
        age = np.where(np.isfinite(age) & (age >= 10) & (age <= 70), age, 0.0)
        gender = np.asarray([{"male": 1, "female": 2}.get(g, 0) for g in joined["gender"]])
        education = cfin_module.code(joined["education"], cfin_module.EDUCATIONS)
        enroll_num = np.asarray([len(self.learner_courses[h]) for h in self.learner], dtype=float)
        learner_raw = np.stack([age, gender, education, cluster, enroll_num], axis=1)[first]
        train_learners = np.unique(self.learner[is_train])
        self.learner_features = data_module.scale(learner_raw, data_module.fit_scaler(learner_raw[train_learners]))

        course_count = len(cache["course_ids"])
        category = np.zeros(course_count)
        category[self.course] = cfin_module.code(joined["category"], CATEGORIES)
        course_raw = np.stack([data_module.train_course_sizes(cache), category], axis=1).astype(float)
        self.course_features = data_module.scale(course_raw, data_module.fit_scaler(course_raw))
        self.course_start = np.zeros(course_count, dtype=np.int64)
        self.course_start[self.course] = self.start
        self.rule = rule

        self.sequence = torch.as_tensor(np.concatenate([
            data_module.action_day_counts(cache[n], CATFHN_ACTIONS) for n in config.SPLITS]), dtype=torch.float32)

    def split_rows(self, name):
        index = config.SPLITS.index(name)
        return np.arange(self.offsets[index], self.offsets[index + 1])

    # Learner-course pairs; `sends` = the learner may send to the course.
    def learner_course_pairs(self):
        learners = np.repeat(np.arange(len(self.learner_courses)), [len(c) for c in self.learner_courses])
        courses = np.concatenate([np.asarray(c) for c in self.learner_courses])
        if self.rule == "temporal":
            sends = self.course_start[courses] == self.learner_time[learners]
        else:
            sends = np.ones(len(courses), dtype=bool)
        return learners, courses, sends


# ---------------------------------------------------------------------------
# 2. Link prediction (linkprediction_models.py)
# ---------------------------------------------------------------------------

class LinkEncoder(nn.Module):
    """EncoderModel with HeteroGNNModel turned hetero by hand (to_hetero, aggr mean:
    one SAGEConv per relation; each node type has one incoming relation)."""

    def __init__(self, learner_count, course_count, learner_dim, course_dim):
        from torch_geometric.nn import SAGEConv

        super().__init__()
        hidden = 16
        self.user_lin = nn.Linear(learner_dim, hidden)
        self.course_lin = nn.Linear(course_dim, hidden)
        self.user_embed = nn.Embedding(learner_count, hidden)
        self.course_embed = nn.Embedding(course_count, hidden)
        self.conv1 = nn.ModuleDict({"to_user": SAGEConv(2 * hidden, 32), "to_course": SAGEConv(2 * hidden, 32)})
        self.conv2 = nn.ModuleDict({"to_user": SAGEConv(2 * hidden, 16), "to_course": SAGEConv(2 * hidden, 16)})

    def forward(self, learner_x, course_x, to_user, to_course):
        user = torch.cat([self.user_embed.weight, self.user_lin(learner_x)], dim=1)
        course = torch.cat([self.course_embed.weight, self.course_lin(course_x)], dim=1)
        user, course = (F.relu(self.conv1["to_user"]((course, user), to_user)),
                        F.relu(self.conv1["to_course"]((user, course), to_course)))
        return (self.conv2["to_user"]((course, user), to_user),
                self.conv2["to_course"]((user, course), to_course))


def message_edges(learners, courses, sends, keep):
    to_user = torch.as_tensor(np.stack([courses[keep], learners[keep]]))
    sending = keep & sends
    to_course = torch.as_tensor(np.stack([learners[sending], courses[sending]]))
    return to_user, to_course


def link_prediction(enrollments, settings, seed, device):
    learners, courses, sends = enrollments.learner_course_pairs()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(learners))
    validation = order[:len(order) // 10]
    rest = order[len(order) // 10:]
    supervision = rest[:int(0.3 * len(rest))]
    passing = np.zeros(len(learners), dtype=bool)
    passing[rest[int(0.3 * len(rest)):]] = True
    to_user, to_course = [e.to(device) for e in message_edges(learners, courses, sends, passing)]
    learner_x = torch.as_tensor(enrollments.learner_features, device=device)
    course_x = torch.as_tensor(enrollments.course_features, device=device)
    learner_count, course_count = len(learner_x), len(course_x)

    def negatives(count):
        return rng.integers(0, learner_count, count), rng.integers(0, course_count, count)

    v_neg = negatives(2 * len(validation))
    v_pairs = (np.concatenate([learners[validation], v_neg[0]]), np.concatenate([courses[validation], v_neg[1]]))
    v_label = np.concatenate([np.ones(len(validation)), np.zeros(len(v_neg[0]))])

    encoder = LinkEncoder(learner_count, course_count, learner_x.shape[1], course_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=settings["link_learning_rate"])
    best_auc, best_state = -1.0, None
    batch = settings["link_batch_size"]
    from sklearn.metrics import roc_auc_score
    for epoch in range(1, settings["link_epochs"] + 1):
        encoder.train()
        rng.shuffle(supervision)
        total = 0.0
        for start in range(0, len(supervision), batch):
            rows = supervision[start:start + batch]
            neg = negatives(3 * len(rows))
            user_rows = torch.as_tensor(np.concatenate([learners[rows], neg[0]]), device=device)
            course_rows = torch.as_tensor(np.concatenate([courses[rows], neg[1]]), device=device)
            target = torch.cat([torch.ones(len(rows)), torch.zeros(len(neg[0]))]).to(device)
            user, course = encoder(learner_x, course_x, to_user, to_course)
            loss = F.binary_cross_entropy_with_logits((user[user_rows] * course[course_rows]).sum(-1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        encoder.eval()
        with torch.no_grad():
            user, course = encoder(learner_x, course_x, to_user, to_course)
            score = (user[torch.as_tensor(v_pairs[0], device=device)]
                     * course[torch.as_tensor(v_pairs[1], device=device)]).sum(-1).cpu().numpy()
        auc = roc_auc_score(v_label, score)
        log("link", f"epoch {epoch}: loss={total / len(supervision):.4f}, link auc={auc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in encoder.state_dict().items()}

    # Embeddings of the best encoder on the full learner-course graph.
    encoder.load_state_dict(best_state)
    encoder.eval()
    every = np.ones(len(learners), dtype=bool)
    full_user, full_course = [e.to(device) for e in message_edges(learners, courses, sends, every)]
    with torch.no_grad():
        user, course = encoder(learner_x, course_x, full_user, full_course)
        links = torch.sigmoid(user @ course.T) >= settings["link_threshold"]
    links = links.cpu().numpy()
    links[learners, courses] = True
    return user.cpu().numpy(), course.cpu().numpy(), links


# ---------------------------------------------------------------------------
# 3. Strong classmates graph and node contexts (graphgeneration/0-1)
# ---------------------------------------------------------------------------

def classmates(enrollments, links, settings, seed):
    rng = np.random.default_rng(seed)
    rows = links[enrollments.learner].astype(np.float32)
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    is_train = np.arange(enrollments.count) < enrollments.offsets[1]
    sources, targets = [], []
    for course in np.unique(enrollments.course):
        members = np.flatnonzero(enrollments.course == course)
        senders = members[is_train[members]]
        if len(senders) == 0:
            continue
        for start in range(0, len(members), 2048):
            receivers = members[start:start + 2048]
            similar = (rows[receivers] @ rows[senders].T) >= settings["similarity"]
            similar &= receivers[:, None] != senders[None, :]
            for row, receiver in enumerate(receivers):
                found = senders[similar[row]]
                if len(found) > settings["max_classmates"]:
                    found = rng.choice(found, settings["max_classmates"], replace=False)
                sources.append(found)
                targets.append(np.full(len(found), receiver))
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    order = np.lexsort((sources, targets))  # CSR of incoming edges per node
    indptr = np.searchsorted(targets[order], np.arange(enrollments.count + 1))
    log("classmates", f"{len(sources):,} edges (train -> any), "
        f"{np.mean(np.diff(indptr) > 0):.1%} of enrollments have classmates")
    return indptr, sources[order]


def contexts(enrollments, user_embedding, course_embedding):
    mean_course_features = np.stack([enrollments.course_features[list(c)].mean(0)
                                     for c in enrollments.learner_courses])
    mean_course_embedding = np.stack([course_embedding[list(c)].mean(0) for c in enrollments.learner_courses])
    learner = enrollments.learner
    original = np.hstack([enrollments.learner_features[learner], mean_course_features[learner]])
    enhanced = np.hstack([user_embedding[learner], mean_course_embedding[learner]])
    return torch.as_tensor(original, dtype=torch.float32), torch.as_tensor(enhanced, dtype=torch.float32)


# NeighborLoader([8, 4]): seeds first, then sampled in-neighbors hop by hop.
def sample(seeds, indptr, indices, fanouts, rng):
    nodes = list(seeds)
    position = {node: i for i, node in enumerate(nodes)}
    frontier = list(seeds)
    sources, targets = [], []
    for fanout in fanouts:
        following = []
        for node in frontier:
            neighbors = indices[indptr[node]:indptr[node + 1]]
            if len(neighbors) > fanout:
                neighbors = rng.choice(neighbors, fanout, replace=False)
            for neighbor in neighbors.tolist():
                if neighbor not in position:
                    position[neighbor] = len(nodes)
                    nodes.append(neighbor)
                    following.append(neighbor)
                sources.append(position[neighbor])
                targets.append(position[node])
        frontier = following
    return np.asarray(nodes), torch.as_tensor(np.asarray([sources, targets], dtype=np.int64).reshape(2, -1))


# ---------------------------------------------------------------------------
# 4. Classifier (train.py)
# ---------------------------------------------------------------------------

def fixed_lgb(original):
    class FixedLGB(original.LGB):
        # models.LGB.forward with `if i == 0` instead of `if i == 1` (see header).
        def forward(self, sub_graph):
            batch_size = sub_graph["batch_size"]
            context = self.context_embed(sub_graph)
            context = self.gnn(context, sub_graph["edge_index"])
            context_output = context[:batch_size]
            input_matrix = sub_graph["seq_feat"][:batch_size]
            input_matrix = input_matrix.view(batch_size, self.week_count, -1, self.activity_num)
            lstm_input = None
            for i in range(self.week_count):
                x = input_matrix[:, i, :, :]
                act_sum_by_day = torch.sum(x, dim=2).view(batch_size, -1, 1)
                x = torch.cat((x, act_sum_by_day), dim=2)
                act_sum_by_action = torch.sum(x, dim=1).view(batch_size, 1, -1)
                x = torch.cat((x, act_sum_by_action), dim=1)
                x = x.view(batch_size, 1, -1)
                if i == 0:
                    lstm_input = x
                else:
                    lstm_input = torch.cat((lstm_input, x), dim=1)
            lstm_output = self.lstm1(lstm_input)
            attention_output = self.self_attention1(lstm_output)
            lstm_ouput2 = self.lstm2(attention_output)
            attention_output2 = self.self_attention2(lstm_ouput2)
            context_output = context_output.repeat_interleave(attention_output2.size(1), 0)
            context_output = context_output.view(batch_size, attention_output2.size(1), -1)
            weighted_sum_input = torch.cat((context_output, attention_output2), dim=2)
            all_feat = self.weighted_sum(weighted_sum_input)
            all_mean_feat = torch.mean(all_feat, dim=1)
            return self.classifier(all_mean_feat)

    return FixedLGB


class Batches:
    def __init__(self, enrollments, graph, original, enhanced, rows, batch_size, shuffle, seed, device):
        self.enrollments, self.graph, self.rows = enrollments, graph, rows
        self.original, self.enhanced = original, enhanced
        self.batch_size, self.shuffle, self.device = batch_size, shuffle, device
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        rows = self.rng.permutation(self.rows) if self.shuffle else self.rows
        for start in range(0, len(rows), self.batch_size):
            seeds = rows[start:start + self.batch_size]
            nodes, edge_index = sample(seeds, *self.graph, (8, 4), self.rng)
            yield {"batch_size": len(seeds),
                   "org_context": self.original[nodes].to(self.device),
                   "enhanced_context": self.enhanced[nodes].to(self.device),
                   "seq_feat": self.enrollments.sequence[nodes].to(self.device),
                   "edge_index": edge_index.to(self.device)}, self.enrollments.label[seeds]


@torch.no_grad()
def predict(model, batches):
    model.eval()
    labels, probabilities = [], []
    for sub_graph, label in batches:
        probabilities.append(torch.sigmoid(model(sub_graph))[:, 0].cpu())
        labels.append(label)
    return np.concatenate(labels).astype(int), torch.cat(probabilities).numpy()


def train_and_test(settings, seed, output_dir, device_name):
    data_module.set_seed(seed)
    device = data_module.resolve_device(device_name)
    original = data_module.import_original("CA-TFHN", "src/models.py", "catfhn_models")
    run_name = data_module.make_run_name(settings, seed)
    log("setup", f"run={run_name}, device={device}, settings={settings}")
    cache = data_module.limit_cache(data_module.load_cache(output_dir), settings["limit"])
    enrollments = Enrollments(cache, settings["user_rule"], settings["clusters"], seed)
    log("setup", f"{len(enrollments.learner_courses):,} learner nodes (rule {settings['user_rule']})")
    user_embedding, course_embedding, links = link_prediction(enrollments, settings, seed, device)
    graph = classmates(enrollments, links, settings, seed)
    original_context, enhanced_context = contexts(enrollments, user_embedding, course_embedding)
    batches = {name: Batches(enrollments, graph, original_context, enhanced_context,
                             enrollments.split_rows(name), settings["batch_size"] if name == "train" else 128,
                             name == "train", seed, device) for name in config.SPLITS}

    model = fixed_lgb(original)(PARAMETERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    checkpoint_path = Path(config.RUNS) / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], {"score": -1.0, "epoch": 0}
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for step, (sub_graph, label) in enumerate(batches["train"], start=1):
            truth = torch.as_tensor(label, dtype=torch.float32, device=device).view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(model(sub_graph), truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(label)
            seen += len(label)
            if step % 100 == 0:
                log("train", f"epoch {epoch}: {seen:,} targets, loss={total / seen:.4f}")
        record = {"epoch": epoch, "loss": total / seen}
        record["validation"] = data_module.epoch_validation(*predict(model, batches["validation"]))
        if record["validation"]["auprc"] > best["score"]:
            best = {"score": record["validation"]["auprc"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "settings": settings, "seed": seed, "epoch": epoch},
                       checkpoint_path)
            log("checkpoint", f"new best val_auprc={best['score']:.4f} at epoch {epoch}")
        history.append(record)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    data_module.write_reports(run_name, settings, seed, history=history, best_epoch=best["epoch"],
                              checkpoint_path=checkpoint_path, validation=predict(model, batches["validation"]),
                              test=predict(model, batches["test"]))


if __name__ == "__main__":
    parser = data_module.base_parser("Train and test CA-TFHN on XuetangX.", DEFAULT_SETTINGS)
    arguments = parser.parse_args()
    settings = data_module.settings_from(arguments, DEFAULT_SETTINGS)
    for seed in arguments.seeds:
        train_and_test(settings, seed, arguments.output_dir, arguments.device)
