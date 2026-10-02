# 12. Baseline SIG-Net (Kim et al., ACM SAC 2024): Student Interaction Graph Network.
#
#   .venvs/signet/bin/python src/12_signet.py --mode both --seeds 1 --user-rule temporal
#
# Original code: baseline/SIG-Net (github.com/Noverse0/SIG-Net, commit 87bc320).
# The model, Multi_RGCN of src/model.py (2 RelGraphConv layers, 11 relations, a
# 2-layer BiLSTM over 3 time windows, MLP), is imported unchanged. The released
# graph builder only reads KDD Cup 2015 and NAVER, so the same subgraphs are
# built here for XuetangX (graph_builder.kddcup_graph, translated):
#
#   one subgraph per enrollment t and per time window w (3 windows):
#     nodes  t, the learner's other enrollments, t's course, every object of the course
#     edges  object <-> course        type 0-3 by object kind (problem, video, discussion, other)
#            t <-> object it used in w type 4-7 by object kind
#            enrollment -> enrollment of the same learner, typed by the sender's
#                                      status: 8 dropout, 9 not dropout. In windows 1, 2
#                                      the status is "no activity in the next window";
#                                      in window 3 it is the true label.
#                                      Edges sent by t carry the receiver's status, so t's
#                                      own status is hidden (as __getitem__ does).
#   features  [node type one-hot ‖ daily event counts of the window, standardized per
#              course ‖ course features] (enrollment / object / course)
#
# What differs from the original, and why:
#   - Windows: KDD has 30 days in 3 windows of 10; XuetangX has 35 days, so the
#     windows are days 0-11, 12-23, 24-34 (12 features each, the last padded with 0).
#   - Course features: KDD counts course modules by category (object.csv); XuetangX
#     has no module list, so: number of video, problem, forum objects seen in train
#     events, and the number of train enrollments.
#   - Protocol (10_baseline_data.py): the learner's enrollments are train
#     enrollments chosen by --user-rule; the original adds every enrollment,
#     including test ones, and types window-3 edges by their true (test) labels.
#     Here a neighbor's true label is only used when it is a train label already
#     known at the target's prediction time: rule "temporal" requires the
#     neighbor's course to have ended by day 35 of the target's course, rule
#     "any" uses every train label. Otherwise the edge gets the extra relation 10
#     ("status unknown"); the model has 11 relations and the original uses 0-9.
#   - Per-course standardization and course statistics are fitted on train.
#   - Each epoch is validated (AUPRC) and the best epoch is kept; the original
#     trains 4 epochs and reports the last on test.
# Hyperparameters of the original main.py: RGCN, 2 layers, hidden 64, BiLSTM,
# dropout 0.3, lr 1e-3, batch 256, 4 epochs, BCE.

from importlib import import_module
from pathlib import Path

import numpy as np
import torch

config = import_module("0_config")
data_module = import_module("10_baseline_data")
log = data_module.log

WINDOWS = ((0, 12), (12, 24), (24, 35))
WINDOW_DAYS = 12
# KDD event codes of the original: problem 0, video 1, discussion 2, other 3.
KDD_KIND = {"assignment": 0, "video": 1, "forum": 2}
KIND_OF_FAMILY = np.asarray([KDD_KIND[family] for family in data_module.OBJECT_FAMILIES])
DROPOUT, NOT_DROPOUT, UNKNOWN = 8, 9, 10
NODE_TYPES = 3
DEFAULT_SETTINGS = {
    "model": "signet",
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 4,
    "num_workers": 4,
}


def csr(keys, values, count):
    order = np.lexsort((values, keys))
    keys, values = keys[order], values[order]
    indptr = np.searchsorted(keys, np.arange(count + 1))
    return indptr, values


# Daily counts standardized per course and day, fitted on the course's train rows.
def window_features(cache):
    train = cache["train"]
    counts = {name: data_module.daily_counts(cache[name]).astype(np.float64) for name in config.SPLITS}
    course_count = len(cache["course_ids"])
    total = np.zeros((course_count, config.OBSERVATION_DAYS))
    square = np.zeros_like(total)
    np.add.at(total, train["course"], counts["train"])
    np.add.at(square, train["course"], counts["train"] ** 2)
    size = data_module.train_course_sizes(cache)[:, None]
    global_mean, global_std = data_module.fit_scaler(counts["train"])
    mean = np.where(size > 0, total / np.maximum(size, 1), global_mean)
    std = np.sqrt(np.maximum(np.where(size > 0, square / np.maximum(size, 1), 0) - mean ** 2, 0))
    std = np.where(size > 0, std, global_std)
    std = np.where(std > 1e-12, std, 1.0)

    features, inactive = {}, {}
    for name in config.SPLITS:
        course = cache[name]["course"]
        scaled = (counts[name] - mean[course]) / std[course]
        windows = np.zeros((len(course), len(WINDOWS), WINDOW_DAYS), dtype=np.float32)
        for w, (first, last) in enumerate(WINDOWS):
            windows[:, w, :last - first] = scaled[:, first:last]
        features[name] = windows
        inactive[name] = np.stack([counts[name][:, first:last].sum(axis=1) == 0 for first, last in WINDOWS], 1)
    return features, inactive


# Objects of every course (seen in train events) and course features.
def course_objects(cache):
    train = cache["train"]
    used = train["event_object"] >= 0
    pairs = np.unique(np.stack([train["course"][train["event_node"][used]], train["event_object"][used]]), axis=1)
    course_count = len(cache["course_ids"])
    indptr, objects = csr(pairs[0], pairs[1], course_count)
    kinds = np.zeros((course_count, 3))
    np.add.at(kinds, pairs[0], np.eye(3)[cache["object_family"][pairs[1]]])
    raw = np.hstack([kinds, data_module.train_course_sizes(cache)[:, None]])
    return (indptr, objects), data_module.scale(raw, data_module.fit_scaler(raw))


# Objects each node used, per window: CSR over node * 3 + window.
def window_objects(split):
    used = split["event_object"] >= 0
    window = np.searchsorted([last for _, last in WINDOWS], split["event_day"][used], side="right")
    keys = split["event_node"][used].astype(np.int64) * len(WINDOWS) + window
    pairs = np.unique(np.stack([keys, split["event_object"][used].astype(np.int64)]), axis=1)
    return csr(pairs[0], pairs[1], len(split["label"]) * len(WINDOWS))


class SigNetData:
    def __init__(self, cache, rule):
        self.cache = cache
        self.rule = rule
        self.features, self.inactive = window_features(cache)
        self.course_objects, self.course_features = course_objects(cache)
        self.window_objects = {name: window_objects(cache[name]) for name in config.SPLITS}
        self.same_user = {name: data_module.same_user_train(cache, cache[name], rule) for name in config.SPLITS}
        self.kind = KIND_OF_FAMILY[cache["object_family"]]
        self.course_dim = self.course_features.shape[1]
        self.input_dim = NODE_TYPES + WINDOW_DAYS + self.course_dim

    # Status relation of train neighbors in window w, seen from target t.
    def neighbor_status(self, neighbors, window, split, target):
        train = self.cache["train"]
        if window < len(WINDOWS) - 1:
            dropped = self.inactive["train"][neighbors, window + 1]
            return np.where(dropped, DROPOUT, NOT_DROPOUT)
        status = np.where(train["label"][neighbors] == 1, DROPOUT, NOT_DROPOUT)
        if self.rule == "temporal":
            known = train["end"][neighbors] <= split["start"][target] + config.OBSERVATION_DAYS
            status = np.where(known, status, UNKNOWN)
        return status

    # Node features, edges (src, dst, type), and target / course positions of
    # the 3 window subgraphs of `target` in split `name`.
    def subgraph_arrays(self, name, target):
        split = self.cache[name]
        course = int(split["course"][target])
        neighbors = data_module.csr_rows(*self.same_user[name], target)
        indptr, objects = self.window_objects[name]
        own = [objects[indptr[target * 3 + w]:indptr[target * 3 + w + 1]] for w in range(len(WINDOWS))]
        course_objects = data_module.csr_rows(*self.course_objects, course)
        all_objects = np.unique(np.concatenate([course_objects, *own]))

        enrollments = 1 + len(neighbors)
        course_node = enrollments
        object_nodes = course_node + 1 + np.arange(len(all_objects))
        node_count = course_node + 1 + len(all_objects)
        kinds = self.kind[all_objects]
        # object <-> course, the same in every window
        base_src = np.concatenate([object_nodes, np.full(len(all_objects), course_node)])
        base_dst = np.concatenate([np.full(len(all_objects), course_node), object_nodes])
        base_type = np.concatenate([kinds, kinds])
        # learner edges: every ordered pair of the learner's enrollments
        members = np.arange(enrollments)
        src, dst = np.meshgrid(members, members, indexing="ij")
        pair = src != dst
        src, dst = src[pair], dst[pair]

        windows = []
        for w in range(len(WINDOWS)):
            features = np.zeros((node_count, self.input_dim), dtype=np.float32)
            features[:enrollments, 0] = 1
            features[0, NODE_TYPES:NODE_TYPES + WINDOW_DAYS] = self.features[name][target, w]
            features[1:enrollments, NODE_TYPES:NODE_TYPES + WINDOW_DAYS] = self.features["train"][neighbors, w]
            features[object_nodes, 1] = 1
            features[course_node, 2] = 1
            features[course_node, NODE_TYPES + WINDOW_DAYS:] = self.course_features[course]

            used = object_nodes[np.searchsorted(all_objects, own[w])]
            used_kind = self.kind[own[w]] + 4
            status = np.concatenate([[UNKNOWN], self.neighbor_status(neighbors, w, split, target)])
            learner_type = np.where(src == 0, status[dst], status[src])
            windows.append({
                "features": features,
                "src": np.concatenate([base_src, np.zeros(len(used), np.int64), used, src]),
                "dst": np.concatenate([base_dst, used, np.zeros(len(used), np.int64), dst]),
                "type": np.concatenate([base_type, used_kind, used_kind, learner_type]),
            })
        return windows, course_node


class SigNetDataset(torch.utils.data.Dataset):
    def __init__(self, data, name, targets):
        self.data, self.name, self.targets = data, name, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        import dgl

        target = int(self.targets[index])
        windows, course_node = self.data.subgraph_arrays(self.name, target)
        graphs = []
        for window in windows:
            graph = dgl.graph((torch.as_tensor(window["src"]), torch.as_tensor(window["dst"])),
                              num_nodes=len(window["features"]))
            graph.ndata["feature"] = torch.as_tensor(window["features"])
            graph.edata["etype"] = torch.as_tensor(window["type"], dtype=torch.int64)
            mark = torch.zeros(graph.num_nodes())
            mark[0], mark[course_node] = 1, 2
            graph.ndata["target"] = mark
            graphs.append(graph)
        label = torch.tensor(float(self.data.cache[self.name]["label"][target]))
        return (*graphs, label)


def collate(items):
    import dgl

    first, second, third, labels = map(list, zip(*items))
    return dgl.batch(first), dgl.batch(second), dgl.batch(third), torch.stack(labels)


def loader(data, name, settings, shuffle, seed):
    targets = np.arange(len(data.cache[name]["label"]))
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        SigNetDataset(data, name, targets), batch_size=settings["batch_size"], shuffle=shuffle,
        collate_fn=collate, num_workers=settings["num_workers"], generator=generator,
        persistent_workers=settings["num_workers"] > 0)


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    labels, probabilities = [], []
    for first, second, third, label in batches:
        output = model(first.to(device), second.to(device), third.to(device), False)
        probabilities.append(output.cpu())
        labels.append(label)
    return torch.cat(labels).numpy().astype(int), torch.cat(probabilities).numpy()


def train_and_test(settings, seed, output_dir, device_name):
    data_module.set_seed(seed)
    import dgl

    dgl.random.seed(seed)
    original = data_module.import_original("SIG-Net", "src/model.py", "signet_model")
    device = data_module.resolve_device(device_name)
    run_name = data_module.make_run_name(settings, seed)
    log("setup", f"run={run_name}, device={device}, settings={settings}")
    cache = data_module.limit_cache(data_module.load_cache(output_dir), settings["limit"])
    data = SigNetData(cache, settings["user_rule"])
    batches = {name: loader(data, name, settings, name == "train", seed) for name in config.SPLITS}

    model = original.Multi_RGCN(settings["num_layers"], data.input_dim, settings["hidden_dim"], True,
                                settings["dropout"], settings["batch_size"], device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    loss_function = torch.nn.BCELoss()
    checkpoint_path = Path(config.RUNS) / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], {"score": -1.0, "epoch": 0}
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for step, (first, second, third, label) in enumerate(batches["train"], start=1):
            output = model(first.to(device), second.to(device), third.to(device), True)
            loss = loss_function(output, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(label)
            seen += len(label)
            if step % 100 == 0:
                log("train", f"epoch {epoch}: {seen:,} targets, loss={total / seen:.4f}")
        record = {"epoch": epoch, "loss": total / seen}
        record["validation"] = data_module.epoch_validation(*predict(model, batches["validation"], device))
        if record["validation"]["auprc"] > best["score"]:
            best = {"score": record["validation"]["auprc"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "settings": settings, "seed": seed, "epoch": epoch},
                       checkpoint_path)
            log("checkpoint", f"new best val_auprc={best['score']:.4f} at epoch {epoch}")
        history.append(record)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    data_module.write_reports(run_name, settings, seed, history=history, best_epoch=best["epoch"],
                              checkpoint_path=checkpoint_path,
                              validation=predict(model, batches["validation"], device),
                              test=predict(model, batches["test"], device))


if __name__ == "__main__":
    parser = data_module.base_parser("Train and test SIG-Net on XuetangX.", DEFAULT_SETTINGS)
    arguments = parser.parse_args()
    settings = data_module.settings_from(arguments, DEFAULT_SETTINGS)
    for seed in arguments.seeds:
        train_and_test(settings, seed, arguments.output_dir, arguments.device)
