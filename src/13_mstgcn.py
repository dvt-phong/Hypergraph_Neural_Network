# 13. Baseline MST-GCN (Scientific Reports 2026): Multi-Scale Spatio-Temporal GCN.
#
#   .venvs/mstgcn/bin/python src/13_mstgcn.py --mode both --seeds 1 --user-rule temporal
#
# Original code: baseline/MST-GCN (github.com/wudongze9/MST-GCN, commit 7350a7e),
# folder MST-GCN/src_xuet (the authors' XuetangX version). The model,
# Multi_MST_GCN of model.py, is imported unchanged; the subgraphs are those of
# xuetangx_graph_builder.XuetangXGraph:
#
#   one heterogeneous subgraph per enrollment t
#     nodes  enrollment: t and the learner's other enrollments
#            object: the objects t used; course: t's course
#     edges  enrollment <-> object (e_observes_o, o_observed_by_e),
#            object <-> course (o_belongs_to_c, c_contains_o)
#   features enrollment  35 daily event counts (raw) ‖ standardized [events, distinct
#                        objects, count of every action]
#            object      standardized count of every action on the object
#            course      standardized [events, enrollments, learners, events per enrollment]
#   As in the released code, the same subgraph is given for the 3 time steps
#   (g1 = g2 = g3), and the learner's other enrollments have no edge (friends.csv
#   does not exist for XuetangX).
#
# What differs from the original, and why:
#   - The original's GraphConv layers refuse nodes without incoming edges, and the
#     learner's other enrollments (and enrollments without objects) have none, so
#     the released code stops with a DGLError. allow_zero_in_degree is switched on
#     after the model is built: such nodes receive zero, which GraphConv's degree
#     clamp already assumes. Nothing else in model.py changes.
#   - Protocol (10_baseline_data.py): the learner's other enrollments are train
#     enrollments by --user-rule (the original uses all, train and test); object
#     and course statistics and the scalers come from train events only (the
#     original pools train and test logs); course start days come from
#     course_info.csv (the original takes the first log time of each course).
#   - Split: the original splits all enrollments 80/20 at random and reports the
#     last of 8 epochs; here the common split, the best epoch on validation AUPRC.
# Hyperparameters of the original main.py: 2 layers, hidden 128, dropout 0.3,
# lr 1e-4, weight decay 1e-5, batch 256, 8 epochs, BCE.

from importlib import import_module
from pathlib import Path

import numpy as np
import torch

config = import_module("0_config")
data_module = import_module("10_baseline_data")
log = data_module.log

DEFAULT_SETTINGS = {
    "model": "mstgcn",
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "epochs": 8,
    "num_workers": 4,
}


class MstGcnData:
    def __init__(self, cache, rule):
        self.cache = cache
        train = cache["train"]
        action_count = len(config.ACTIONS)

        # Enrollment: raw daily counts ‖ standardized aggregates.
        aggregates = {}
        for name in config.SPLITS:
            split = cache[name]
            used = split["event_object"] >= 0
            pairs = np.unique(np.stack([split["event_node"][used], split["event_object"][used]]), axis=1)
            distinct = np.bincount(pairs[0], minlength=len(split["label"]))
            actions = data_module.action_counts(split)
            aggregates[name] = np.hstack([actions.sum(1, keepdims=True), distinct[:, None], actions]).astype(float)
        scaler = data_module.fit_scaler(aggregates["train"])
        self.enrollment = {name: np.hstack([data_module.daily_counts(cache[name]),
                                            data_module.scale(aggregates[name], scaler)]).astype(np.float32)
                           for name in config.SPLITS}

        # Object: action counts on the object in train events.
        used = train["event_object"] >= 0
        object_actions = np.zeros((len(cache["object_keys"]), action_count))
        np.add.at(object_actions, (train["event_object"][used], train["event_action"][used]), 1)
        seen = np.unique(train["event_object"][used])
        self.object = data_module.scale(object_actions, data_module.fit_scaler(object_actions[seen]))

        # Course: events, enrollments, learners, events per enrollment (train).
        course_count = len(cache["course_ids"])
        events = np.bincount(train["course"][train["event_node"]], minlength=course_count)
        enrollments = data_module.train_course_sizes(cache)
        learners = np.asarray([len(np.unique(train["user_id"][train["course"] == c])) for c in range(course_count)])
        raw = np.stack([events, enrollments, learners, events / np.maximum(enrollments, 1)], 1).astype(float)
        self.course = data_module.scale(raw, data_module.fit_scaler(raw))

        # Objects each node used (CSR), and the learner's train enrollments.
        self.objects = {}
        for name in config.SPLITS:
            split = cache[name]
            used = split["event_object"] >= 0
            pairs = np.unique(np.stack([split["event_node"][used].astype(np.int64),
                                        split["event_object"][used].astype(np.int64)]), axis=1)
            self.objects[name] = (np.searchsorted(pairs[0], np.arange(len(split["label"]) + 1)), pairs[1])
        self.same_user = {name: data_module.same_user_train(cache, cache[name], rule) for name in config.SPLITS}
        self.dims = {"enrollment": self.enrollment["train"].shape[1], "object": self.object.shape[1],
                     "course": self.course.shape[1]}

    def subgraph_arrays(self, name, target):
        neighbors = data_module.csr_rows(*self.same_user[name], target)
        objects = data_module.csr_rows(*self.objects[name], target)
        local = np.arange(len(objects))
        zeros = np.zeros(len(objects), dtype=np.int64)
        return {
            "edges": {
                ("enrollment", "e_observes_o", "object"): (zeros, local),
                ("object", "o_observed_by_e", "enrollment"): (local, zeros),
                ("object", "o_belongs_to_c", "course"): (local, zeros),
                ("course", "c_contains_o", "object"): (zeros, local),
            },
            "counts": {"enrollment": 1 + len(neighbors), "object": len(objects), "course": 1},
            "enrollment": np.vstack([self.enrollment[name][target:target + 1], self.enrollment["train"][neighbors]]),
            "object": self.object[objects],
            "course": self.course[self.cache[name]["course"][target]][None],
        }


class MstGcnDataset(torch.utils.data.Dataset):
    def __init__(self, data, name):
        self.data, self.name = data, name

    def __len__(self):
        return len(self.data.cache[self.name]["label"])

    def __getitem__(self, target):
        import dgl

        arrays = self.data.subgraph_arrays(self.name, target)
        graph = dgl.heterograph({relation: (torch.as_tensor(src), torch.as_tensor(dst))
                                 for relation, (src, dst) in arrays["edges"].items()},
                                num_nodes_dict=arrays["counts"])
        for node_type in ("enrollment", "object", "course"):
            graph.nodes[node_type].data["feature"] = torch.as_tensor(arrays[node_type], dtype=torch.float32)
            graph.nodes[node_type].data["target"] = torch.zeros(arrays["counts"][node_type], dtype=torch.long)
        graph.nodes["enrollment"].data["target"][0] = 1
        graph.nodes["course"].data["target"][0] = 2
        label = torch.tensor(float(self.data.cache[self.name]["label"][target]))
        return graph, graph, graph, label


def collate(items):
    import dgl

    graphs, _, _, labels = map(list, zip(*items))
    batched = dgl.batch(graphs)
    return batched, batched, batched, torch.stack(labels)


def loader(data, name, settings, shuffle, seed):
    return torch.utils.data.DataLoader(
        MstGcnDataset(data, name), batch_size=settings["batch_size"], shuffle=shuffle, collate_fn=collate,
        num_workers=settings["num_workers"], generator=torch.Generator().manual_seed(seed),
        persistent_workers=settings["num_workers"] > 0)


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    labels, probabilities = [], []
    for first, _, _, label in batches:
        graph = first.to(device)
        probabilities.append(model(graph, graph, graph, False).cpu())
        labels.append(label)
    return torch.cat(labels).numpy().astype(int), torch.cat(probabilities).numpy()


def train_and_test(settings, seed, output_dir, device_name):
    data_module.set_seed(seed)
    import dgl

    dgl.random.seed(seed)
    original = data_module.import_original("MST-GCN", "MST-GCN/src_xuet/model.py", "mstgcn_model")
    device = data_module.resolve_device(device_name)
    run_name = data_module.make_run_name(settings, seed)
    log("setup", f"run={run_name}, device={device}, settings={settings}")
    cache = data_module.limit_cache(data_module.load_cache(output_dir), settings["limit"])
    data = MstGcnData(cache, settings["user_rule"])
    batches = {name: loader(data, name, settings, name == "train", seed) for name in config.SPLITS}

    model = original.Multi_MST_GCN(
        num_layers=settings["num_layers"], enroll_in_feats=data.dims["enrollment"],
        object_in_feats=data.dims["object"], course_in_feats=data.dims["course"],
        h_feats=settings["hidden_dim"], bi=True, dropout=settings["dropout"],
        batch_size=settings["batch_size"], device=device).to(device)
    for module in model.modules():
        if isinstance(module, dgl.nn.pytorch.GraphConv):
            module._allow_zero_in_degree = True
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    loss_function = torch.nn.BCELoss()
    checkpoint_path = Path(config.RUNS) / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], {"score": -1.0, "epoch": 0}
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for step, (graph, _, _, label) in enumerate(batches["train"], start=1):
            graph = graph.to(device)
            loss = loss_function(model(graph, graph, graph, True), label.to(device))
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
    parser = data_module.base_parser("Train and test MST-GCN on XuetangX.", DEFAULT_SETTINGS)
    arguments = parser.parse_args()
    settings = data_module.settings_from(arguments, DEFAULT_SETTINGS)
    for seed in arguments.seeds:
        train_and_test(settings, seed, arguments.output_dir, arguments.device)
