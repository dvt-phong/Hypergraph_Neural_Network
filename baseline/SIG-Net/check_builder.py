# Check of the XuetangX SIG-Net graph builder (code/src/xuetangx_graph_builder.py).
#
#   python baseline/SIG-Net/check_builder.py [--learners 600] [--split-seed 1]
#
# Takes the enrollments of a random sample of learners, builds the graph arrays with the vectorized
# build_xuetangx_arrays and with a literal loop translation of kddcup_graph.process (iterrows and
# DataFrame filters, as in code/src/graph_builder.py), and checks that edges, edge types, student
# edge ids, node features and labels are identical. Needs pandas, numpy, scikit-learn, torch (no DGL).

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "code" / "src"))
from xuetangx_graph_builder import ACTION_EVENT, WINDOW, ACTIVITY, COURSE_CATE_LIST, add_days_and_event, build_xuetangx_arrays


# Literal translation of kddcup_graph.process (loops kept), same node ids as the fast builder.
def loop_reference(log_all, truth_all, course_info, split, fast):
    log_all = log_all.rename(columns={'enroll_id': 'enrollment_id'})
    truth_all = truth_all.rename(columns={'enroll_id': 'enrollment_id'}).sort_values('enrollment_id').reset_index(drop=True)
    start = course_info.set_index('course_id')['start'].str[:10]
    log_all['days'] = (pd.to_datetime(log_all['time'].str[:10]) - pd.to_datetime(log_all['course_id'].map(start))).dt.days
    log_all['event'] = log_all['action'].map(ACTION_EVENT)
    all_clicksum = log_all[['enrollment_id', 'days', 'object', 'event', 'time']].groupby(['enrollment_id', 'event', 'days', 'object']).count().reset_index()
    all_clicksum.rename(columns={'time': 'click_sum'}, inplace=True)
    windows = [list(range(w * WINDOW, (w + 1) * WINDOW)) for w in range(3)]
    activity_log_all = log_all[log_all['event'].isin(ACTIVITY)]
    interactions = []
    for days in windows:
        w = activity_log_all[activity_log_all['days'].isin(days)]
        w = pd.merge(w, w.groupby(['enrollment_id', 'object']).count().reset_index()[['enrollment_id', 'object', 'time']].rename(columns={'time': 'clicksum'}), how='left')
        interactions.append(w[['enrollment_id', 'event', 'object', 'clicksum']].groupby(['enrollment_id', 'event', 'object', 'clicksum']).nunique().reset_index())
    enroll_node_id, object_node_id, course_node_id = fast['enroll_node_id'], fast['object_node_id'], fast['course_node_id']
    enroll_list = list(enroll_node_id)
    course_list = list(course_node_id)
    enroll_truth = {}
    for i in enroll_list:
        enroll_truth[i] = truth_all[truth_all['enrollment_id'] == i]['truth'].values[0]
    object_info = pd.merge(activity_log_all, truth_all[['enrollment_id', 'truth']], how='left')
    object_info = object_info[['course_id', 'object', 'event']].groupby(['course_id', 'object', 'event']).nunique().reset_index()

    enroll_feature = {}
    labels = []
    for enroll_id in enroll_list:
        feature = np.zeros(3 * WINDOW)
        for index, datarow in all_clicksum[all_clicksum['enrollment_id'] == enroll_id].iterrows():
            feature[datarow['days']] = datarow['click_sum']
        enroll_feature[enroll_id] = feature
        labels.append([int(sum(feature[w * WINDOW:(w + 1) * WINDOW]) == 0) for w in range(3)] + [enroll_truth[enroll_id]])
    labels_df = pd.DataFrame(labels, index=enroll_list)

    e_in_course = defaultdict(list)
    course_feature = {}
    cate_df = object_info.groupby(['course_id', 'event']).size().reset_index(name='count')
    for course in course_list:
        for enroll in truth_all[truth_all['course_id'] == course]['enrollment_id']:
            e_in_course[course_node_id[course]].append(enroll_node_id[enroll])
        feature = np.zeros(len(COURSE_CATE_LIST) + 1)
        for index, datarow in cate_df[cate_df['course_id'] == course].iterrows():
            feature[int(datarow['event'])] = datarow['count']
        feature[-1] = len(e_in_course[course_node_id[course]])
        course_feature[course_node_id[course]] = feature

    node_feature = {}
    for course in course_list:
        rows = truth_all[truth_all['course_id'] == course]['enrollment_id'].tolist()
        train_rows = [e for e in rows if split[e] == 'train']
        x = np.array([enroll_feature[e] for e in rows])
        if train_rows:
            x = StandardScaler().fit(np.array([enroll_feature[e] for e in train_rows])).transform(x)
        for e, row in zip(rows, x):
            node_feature[e] = row
    course_matrix = StandardScaler().fit_transform(np.array([course_feature[course_node_id[c]] for c in course_list]))

    src_node, dst_node, etype = [], [], []
    for index, data_row in object_info.iterrows():
        src_node.append(object_node_id[data_row['object']])
        dst_node.append(course_node_id[data_row['course_id']])
        etype.append(int(data_row['event']))
    graphs = []
    for interaction in interactions:
        s, d, t = src_node.copy(), dst_node.copy(), etype.copy()
        for index, data_row in interaction.iterrows():
            s.append(enroll_node_id[data_row['enrollment_id']])
            d.append(object_node_id[data_row['object']])
            t.append(int(data_row['event']) + 4)
        graphs.append({'src': s + d, 'dst': d + s, 'etype': t + t, 'eid': defaultdict(list)})
    for student in sorted(set(truth_all['username'])):
        student_list_df = truth_all[truth_all['username'] == student]
        student_list = list(map(lambda x: enroll_node_id[x], student_list_df['enrollment_id'].tolist()))
        truths = [list(map(lambda x: labels_df.loc[x].values[k], student_list_df['enrollment_id'].tolist())) for k in (1, 2, 3)]
        in_train = [split[e] == 'train' for e in student_list_df['enrollment_id'].tolist()]
        n = len(student_list)
        for a in range(n):
            for b in range(n):
                if a != b:
                    for g, graph in enumerate(graphs):
                        kind = 8 if truths[g][a] == 1 else 9
                        if g == 2 and not in_train[a]:
                            kind = 10
                        graph['eid'][student_list[a]].append(len(graph['src']))
                        graph['src'].append(student_list[a])
                        graph['dst'].append(student_list[b])
                        graph['etype'].append(kind)
    return graphs, node_feature, course_matrix, labels_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the vectorized XuetangX builder with a loop translation.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "processed" / "baseline_raw")
    parser.add_argument("--learners", type=int, default=600)
    parser.add_argument("--split-seed", type=int, default=1)
    arguments = parser.parse_args()

    enrollments = pd.read_csv(arguments.raw_dir / "prediction_log" / "enrollments.csv")
    learners = np.random.default_rng(0).choice(enrollments['username'].unique(), arguments.learners, replace=False)
    enrollments = enrollments[enrollments['username'].isin(learners)]
    keep = set(enrollments['enroll_id'])
    truth_all = pd.read_csv(arguments.raw_dir / "prediction_log" / "truth.csv").merge(enrollments, on='enroll_id')
    log_all = pd.concat([chunk[chunk['enroll_id'].isin(keep)] for chunk in pd.read_csv(
        arguments.raw_dir / "prediction_log" / "log.csv", usecols=['enroll_id', 'username', 'course_id', 'action', 'object', 'time'],
        chunksize=5_000_000)])
    course_info = pd.read_csv(arguments.raw_dir / "course_info.csv", usecols=['course_id', 'start'])
    split = pd.read_csv(arguments.raw_dir / "splits" / f"learner_s{arguments.split_seed}.csv", index_col='enroll_id')['split']
    print(f"sample: {len(learners)} learners, {len(truth_all)} enrollments, {len(log_all):,} log rows", flush=True)

    fast = build_xuetangx_arrays(log_all.copy(), truth_all.copy(), course_info, split)
    graphs, node_feature, course_matrix, labels_df = loop_reference(log_all.copy(), truth_all.copy(), course_info, split, fast)

    ok = True
    for g in range(3):
        for key in ('src', 'dst', 'etype'):
            same = np.array_equal(np.asarray(graphs[g][key]), fast['graphs'][g][key])
            ok &= same
            print(f"graph {g + 1} {key}: {'identical' if same else 'DIFFERENT'} ({len(graphs[g][key]):,} edges)")
        same = {k: list(v) for k, v in graphs[g]['eid'].items()} == {k: list(v) for k, v in fast['student_eid'][g].items()}
        ok &= same
        print(f"graph {g + 1} student edge ids: {'identical' if same else 'DIFFERENT'}")
        third = np.asarray(graphs[g]['etype'])
        print(f"graph {g + 1} edge types: " + ", ".join(f"{t}:{int((third == t).sum())}" for t in range(11)))
    enroll_ids = list(fast['enroll_node_id'])
    n_enroll = len(enroll_ids)
    loop_features = np.array([node_feature[e] for e in enroll_ids])
    for w in range(3):
        same = np.allclose(fast['graphs'][w]['feature'][:n_enroll, 3:3 + WINDOW], loop_features[:, w * WINDOW:(w + 1) * WINDOW])
        ok &= same
        print(f"enrollment features window {w + 1}: {'identical' if same else 'DIFFERENT'}")
    same = np.allclose(fast['graphs'][0]['feature'][-len(course_matrix):, 3 + WINDOW:], course_matrix)
    ok &= same
    print(f"course features: {'identical' if same else 'DIFFERENT'}")
    same = np.array_equal(fast['labels'], labels_df.loc[enroll_ids].to_numpy().astype(float))
    ok &= same
    print(f"labels: {'identical' if same else 'DIFFERENT'}")
    # the loader keeps only rows with an object (days/event precomputed): same arrays
    loaded = add_days_and_event(log_all[log_all['object'].notna()].copy(), course_info)[['enroll_id', 'course_id', 'object', 'days', 'event']]
    light = build_xuetangx_arrays(loaded, truth_all.copy(), course_info, split)
    same = all(np.array_equal(light['graphs'][g][k], fast['graphs'][g][k]) for g in range(3) for k in ('src', 'dst', 'etype', 'feature'))
    same &= np.array_equal(light['labels'], fast['labels'])
    ok &= same
    print(f"rows with an object only ({len(loaded):,} of {len(log_all):,}): {'identical' if same else 'DIFFERENT'}")
    third = fast['graphs'][2]
    student = third['etype'] >= 8
    leaked = int(((third['etype'] != 10) & student & ~np.isin(third['src'], [fast['enroll_node_id'][e] for e in enroll_ids if split[e] == 'train'])).sum())
    ok &= leaked == 0
    print(f"student edges of graph 3 from a validation/test source with a label type: {leaked}")
    print("CHECK PASSED" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)
