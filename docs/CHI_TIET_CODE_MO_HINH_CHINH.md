# Chi tiết code mô hình chính (`src/0` → `src/10`)

Ngày lập: 2026-10-11, theo code ở commit `0fa817a` (đã sửa `RESULT_COLUMNS`). Số dòng ghi theo file hiện tại; nếu sửa code thì số dòng có thể lệch.

Số liệu thật lấy từ `data/processed/simple` ở máy em (đã chạy lại `3_features.py` ngày 2026-10-11).

**Cách dùng để in và đối chiếu:** mỗi mục 1–11 có bảng tóm tắt, rồi tới phần **"Mã nguồn có chú giải"**: code được **chép nguyên văn bằng script** từ file thật, kèm số dòng; mỗi đoạn có một câu giải thích ngay phía trên. Script đã kiểm tra mỗi dòng của mỗi file xuất hiện đúng một lần, không sót. Dòng đầu phần đó ghi md5 của file; nếu md5 của file hiện tại (`certutil -hashfile <file> MD5` trên Windows, `md5sum <file>` trên Linux) khác với md5 ghi trong tài liệu, nghĩa là code đã đổi và tài liệu cần sinh lại.

---

## 0. Nhìn nhanh

### 0.1 Luồng chạy

```
                    data/raw/xuetangx/
   prediction_data.tar.gz   user_info.csv   course_info.csv
                │
   [1_download] │ (chỉ tải nếu chưa có)
                ▼
   [2_preprocess]  lọc 22 action, ngày 0–34; chia train/val/test
                │
                ▼  data/processed/simple/{train,validation,test}.csv   (1 dòng = 1 sự kiện)
                │
        ┌───────┴──────────────────────┐
        ▼                              ▼
   [3_features]                   [4_hypergraph]
   X.npy mỗi split (89 cột)       hypergraph.npz (H, nhãn, split)
        └───────┬──────────────────────┘
                ▼
   [5_graph_data] ghép X, H; áp kịch bản (bỏ family / bỏ cột; thêm self-loop)
                ▼
   [6_hgnn] + [7_mlp] → [8_model]  (định nghĩa mô hình)
                ▼
   [9_train]  train full-batch, early stopping theo val AUC → outputs/results.csv
                ▼
   [10_summary] → outputs/summary.csv, outputs/ket_qua.xlsx
```

### 0.2 Lệnh chạy (từ thư mục gốc project)

```bash
python src/1_download.py
python src/2_preprocess.py          # BỎ QUA nếu 3 file CSV đã có → muốn làm lại phải xóa trước
python src/3_features.py            # luôn ghi đè X.npy
python src/4_hypergraph.py          # luôn ghi đè hypergraph.npz
python src/9_train.py --scenario M  # hoặc --scenario all ; mặc định 5 seed
python src/10_summary.py
```

### 0.3 Bảng file: đầu vào → đầu ra

| File | Dòng | Đầu vào | Đầu ra | Ghi đè? |
|---|---|---|---|---|
| `0_config.py` | 146 | — | Hằng số cho mọi file | — |
| `1_download.py` | 34 | URL trong config | `data/raw/xuetangx/` (3 file) | Không, bỏ qua file đã có |
| `2_preprocess.py` | 233 | 3 file raw | `simple/{train,validation,test}.csv` | **Không**, bỏ qua nếu đủ 3 file |
| `3_features.py` | 272 | 3 file CSV ở trên | `simple/{split}/X.npy`, `simple/feature_names.csv` | Có |
| `4_hypergraph.py` | 157 | 3 file CSV ở trên | `simple/hypergraph.npz` | Có |
| `5_graph_data.py` | 95 | `X.npy` × 3, `hypergraph.npz` | dict trong bộ nhớ | — |
| `6_hgnn.py` | 76 | graph dict, X | nhánh HGNN `z_g` | — |
| `7_mlp.py` | 23 | X | nhánh MLP `z_s` | — |
| `8_model.py` | 54 | X, graph | logit mỗi node | — |
| `9_train.py` | 265 | như 5 | `outputs/results.csv` (1 dòng / kịch bản / seed) | **Nối thêm** dòng |
| `10_summary.py` | 102 | `outputs/results.csv` | `outputs/summary.csv`, `outputs/ket_qua.xlsx` | Có |

### 0.4 Số liệu thật của dữ liệu hiện tại

| Đại lượng | Giá trị |
|---|---|
| Số node (enrollment) | 225,642 = train 126,354 + val 31,589 + test 67,699 |
| Tỉ lệ dropout | train 0.7582, val 0.7601, test 0.7580 |
| Hyperedge Course | 247 (kích thước 244 – 670 (trung vị) – 4,253) |
| Hyperedge Object | 22,421 (2 – 44 – 3,338) |
| Hyperedge User | 64,672 (2 – 2 – 200) |
| Tổng membership (số ô 1 trong H, chưa tính self-loop) | 3,222,033 |
| Bậc node (chưa tính self-loop) | 1 – 5 (trung vị) – 354. Mọi node có bậc ≥ 1, vì khóa nào cũng có ≥ 244 người |
| X | float32, 89 cột |

### 0.5 Ký hiệu dùng trong tài liệu

| Ký hiệu | Nghĩa | Kích thước |
|---|---|---|
| N | số node = 225,642 | |
| E | số hyperedge (sau khi áp kịch bản, kể cả self-loop) | |
| X | ma trận feature | N × 89 (hoặc ít cột hơn ở F1–F3) |
| H | ma trận liên thuộc: h(v,e) = 1 nếu node v thuộc hyperedge e | N × E (sparse) |
| f(e) | family của hyperedge e: course / object / user / self_loop | |
| θ_f, w_f | tham số và trọng số family: w_f = softplus(θ_f) | 4 |
| w_e | trọng số hyperedge = w_f(e) | E |
| d(v) | bậc có trọng số = Σ_e h(v,e)·w_e | N |
| δ(e) | số node trong e | E |

---

## 1. `0_config.py` — Hằng số

**Vai trò:** chứa mọi hằng số. Muốn đổi đường dẫn, split, danh sách action, kịch bản hay siêu tham số thì **chỉ sửa ở đây**.

| Dòng | Tên | Giá trị / ý nghĩa | File dùng |
|---|---|---|---|
| 7–11 | `ROOT, RAW, PROCESSED, OUTPUTS, RESULTS_CSV` | Đường dẫn: `data/raw/xuetangx`, `data/processed/simple`, `outputs`, `outputs/results.csv` | tất cả |
| 12 | `SEEDS` | (1, 11, 111, 1111, 11111) | 9 |
| 16–23 | `DOWNLOAD_FILES` | 3 URL moocdata | 1 |
| 27 | `SPLITS` | ("train", "validation", "test"). **Thứ tự này = mã split 0, 1, 2** trong `hypergraph.npz` | 2, 3, 4, 5 |
| 28 | `SPLIT_SEED` | 1, seed để xáo train chính thức trước khi cắt 80/20 | 2 |
| 29 | `TRAIN_RATIO` | 0.80 | 2 |
| 30 | `OBSERVATION_DAYS` | 35, nghĩa là chỉ giữ ngày 0–34 tính từ ngày mở khóa | 2, 3 |
| 31–47 | `ACTION_GROUPS` | 4 nhóm: video (5), assignment (6), forum (5), web_page (6) = 22 action | 2, 3, 4 |
| 49–53 | `ACTIONS` | Tuple 22 action theo thứ tự nhóm. **Thứ tự này = thứ tự cột 37–58 của X** | 2, 3 |
| 54–55 | `TRUTH_FILES, LOG_FILES` | Tên file bên trong tar.gz | 2 |
| 59 | `GENDERS` | ("female", "male") | 3 |
| 60–62 | `EDUCATIONS` | 7 mức | 3 |
| 63–67 | `CATEGORIES` | 17 loại khóa học | 3 |
| 68 | `MISSING_VALUES` | Chuỗi coi như thiếu: "", "na", "n/a", "none", "null", "no data", "-" | 3, 4 |
| 69–71 | `AGE_MIN, AGE_MAX, AGE_MISSING` | 10, 70, 0 | 3 |
| 75–80 | Chỉ số khối hành vi | ngày 0–34 → cột 0–34; `TOTAL_EVENTS_INDEX` = 35; `OBJECT_COUNT_INDEX` = 36; `ACTION_FEATURE_START` = 37; `BEHAVIOR_FEATURE_COUNT` = 59 | 3 |
| 82–87 | Chỉ số khối người học (đánh số trong khối) | gender 0–2, education 3–10, `AGE_FEATURE_INDEX` = 11, tổng 12 | 3 |
| 89–91 | Khối khóa học | category 0–17 (17 + missing) = 18 | 3 |
| 93–95 | Vị trí khối trong X | `USER_FEATURE_START` = 59, `COURSE_FEATURE_START` = 71, `TOTAL_FEATURE_COUNT` = 89 | 3, baseline/ML |
| 99–104 | `OBJECT_ACTIONS` | action → nhóm, **chỉ video / assignment / forum** (16 action). web_page không tạo hyperedge Object | 4 |
| 106 | `EDGE_FAMILIES` | ("course", "object", "user", "self_loop"). **Thứ tự = mã family 0–3** | 4, 5, 6, 8 |
| 110–113 | `scenario(...)` | Tạo dict kịch bản: families, features, hgnn_layers, use_mlp, learn_w | 9, 10 |
| 116–130 | `SCENARIOS` | M, A1–A4 (bỏ 1 family), W1 (W = I), F1–F3 (bớt cột), L1 (1 layer), H (bỏ MLP), X1 (chỉ self-loop, tùy chọn) | 9, 10 |
| 132 | `SCENARIOS_ALL` | Các kịch bản chạy khi `--scenario all` (không có X1) | 9 |
| 136 | `THRESHOLD` | 0.5 | 9 |
| 137–146 | `TRAIN` | hidden 128, dropout 0.5, lr 1e-3, weight_decay 5e-4, lr của W 0.05, tối đa 1000 epoch, chấm val mỗi 5 epoch, patience 40 (**tính theo số lần chấm**, xem mục 10) | 9 |

### Mã nguồn có chú giải

File `src/0_config.py`: 146 dòng, md5 `628634c222a3d2ba854e417dfd33ad24`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–5.** Tiêu đề file và import `Path` (xử lý đường dẫn không phụ thuộc Windows/Linux).

```python
  1 | # 0. Setting hyperparameter and config paramater
  2 | 
  3 | from pathlib import Path
  4 | 
  5 | 
```

**Dòng 6–14.** Đường dẫn gốc của project và 5 seed. `ROOT` = thư mục cha của `src/`, nên chạy từ đâu cũng đúng. `RESULTS_CSV` là file mà `9_train.py` nối thêm kết quả.

```python
  6 | # Project paths and experiment seeds.
  7 | ROOT = Path(__file__).resolve().parents[1]
  8 | RAW = ROOT / "data" / "raw" / "xuetangx"
  9 | PROCESSED = ROOT / "data" / "processed" / "simple"
 10 | OUTPUTS = ROOT / "outputs"
 11 | RESULTS_CSV = OUTPUTS / "results.csv"
 12 | SEEDS = (1, 11, 111, 1111, 11111)
 13 | 
 14 | 
```

**Dòng 15–25.** 3 file tải về và URL tương ứng (dùng ở `1_download.py`).

```python
 15 | # Used by 1_download.py.
 16 | DOWNLOAD_FILES = {
 17 |     "prediction_data.tar.gz":
 18 |         "https://lfs.aminer.cn/misc/moocdata/data/prediction_data.tar.gz",
 19 |     "user_info.csv":
 20 |         "https://lfs.aminer.cn/misc/moocdata/data/user_info.csv",
 21 |     "course_info.csv":
 22 |         "https://lfs.aminer.cn/misc/moocdata/data/course_info.csv",
 23 | }
 24 | 
 25 | 
```

**Dòng 26–30.** Thứ tự `SPLITS` = mã split 0/1/2 trong `hypergraph.npz`. `SPLIT_SEED = 1`, `TRAIN_RATIO = 0.8`: cắt train chính thức thành train/val. `OBSERVATION_DAYS = 35`: chỉ giữ ngày 0–34.

```python
 26 | # Used by 2_preprocess.py.
 27 | SPLITS = ("train", "validation", "test")
 28 | SPLIT_SEED = 1
 29 | TRAIN_RATIO = 0.80
 30 | OBSERVATION_DAYS = 35
```

**Dòng 31–47.** 22 action chia 4 nhóm. Nhóm quyết định hyperedge Object (dòng 98–104); thứ tự quyết định thứ tự cột 37–58 của X.

```python
 31 | ACTION_GROUPS = {
 32 |     "video": (
 33 |         "seek_video", "play_video", "pause_video", "stop_video", "load_video"
 34 |     ),
 35 |     "assignment": (
 36 |         "problem_get", "problem_check", "problem_save", "reset_problem",
 37 |         "problem_check_correct", "problem_check_incorrect",
 38 |     ),
 39 |     "forum": (
 40 |         "create_thread", "create_comment", "delete_thread", "delete_comment",
 41 |         "close_forum",
 42 |     ),
 43 |     "web_page": (
 44 |         "click_info", "click_courseware", "click_about", "click_forum",
 45 |         "click_progress", "close_courseware",
 46 |     ),
 47 | }
```

**Dòng 48–57.** Trải 4 nhóm thành tuple 22 action (giữ thứ tự nhóm). Tên file nhãn và log bên trong `prediction_data.tar.gz`.

```python
 48 | # All 22 actions in one tuple, group by group.
 49 | action_list = []
 50 | for group_name in ACTION_GROUPS:
 51 |     for action in ACTION_GROUPS[group_name]:
 52 |         action_list.append(action)
 53 | ACTIONS = tuple(action_list)
 54 | TRUTH_FILES = ("train_truth.csv", "test_truth.csv")
 55 | LOG_FILES = ("train_log.csv", "test_log.csv")
 56 | 
 57 | 
```

**Dòng 58–72.** Danh sách giá trị hợp lệ cho one-hot (2 giới tính, 7 học vấn, 17 loại khóa học), các chuỗi được coi là thiếu, và luật tuổi [10, 70], thiếu → 0.

```python
 58 | # Used by 3_features.py
 59 | GENDERS = ("female", "male")
 60 | EDUCATIONS = (
 61 |     "Associate", "Bachelor's", "Doctorate", "High", "Master's", "Middle", "Primary"
 62 | )
 63 | CATEGORIES = (
 64 |     "art", "biology", "business", "chemistry", "computer", "economics",
 65 |     "education", "electrical", "engineering", "foreign language", "history",
 66 |     "literature", "math", "medicine", "philosophy", "physics", "social science",
 67 | )
 68 | MISSING_VALUES = ("", "na", "n/a", "none", "null", "no data", "-")
 69 | AGE_MIN = 10
 70 | AGE_MAX = 70
 71 | AGE_MISSING = 0
 72 | 
```

**Dòng 73–81.** Vị trí cột trong khối hành vi: ngày 0–34 → cột 0–34; cột 35 = tổng sự kiện; cột 36 = số object khác nhau; cột 37–58 = 22 action. `BEHAVIOR_FEATURE_COUNT` = 59.

```python
 73 | # X = [behavior | user (gender, education, age) | course (category)]
 74 | # 35 daily event counts | total events, distinct objects, 22 action counts
 75 | DAY_FEATURE_COUNT = OBSERVATION_DAYS
 76 | TOTAL_EVENTS_INDEX = DAY_FEATURE_COUNT                 # events in days 0–34
 77 | OBJECT_COUNT_INDEX = TOTAL_EVENTS_INDEX + 1            # distinct objects used in days 0–34
 78 | ACTION_FEATURE_START = OBJECT_COUNT_INDEX + 1
 79 | ACTION_FEATURE_COUNT = len(ACTIONS)
 80 | BEHAVIOR_FEATURE_COUNT = ACTION_FEATURE_START + ACTION_FEATURE_COUNT
 81 | 
```

**Dòng 82–92.** Vị trí cột **bên trong** khối người học (gender 0–2, education 3–10, age 11; tổng 12) và khối khóa học (category 0–17; tổng 18).

```python
 82 | GENDER_FEATURE_START = 0
 83 | GENDER_FEATURE_COUNT = len(GENDERS) + 1
 84 | EDUCATION_FEATURE_START = GENDER_FEATURE_START + GENDER_FEATURE_COUNT
 85 | EDUCATION_FEATURE_COUNT = len(EDUCATIONS) + 1
 86 | AGE_FEATURE_INDEX = EDUCATION_FEATURE_START + EDUCATION_FEATURE_COUNT  # user[:, 11] -> final X[:, 70], z-score
 87 | USER_FEATURE_COUNT = AGE_FEATURE_INDEX + 1
 88 | 
 89 | CATEGORY_FEATURE_START = 0
 90 | CATEGORY_FEATURE_COUNT = len(CATEGORIES) + 1
 91 | COURSE_FEATURE_COUNT = CATEGORY_FEATURE_COUNT
 92 | 
```

**Dòng 93–97.** Vị trí khối trong X: người học bắt đầu ở cột 59, khóa học ở cột 71; tổng **89** cột.

```python
 93 | USER_FEATURE_START = BEHAVIOR_FEATURE_COUNT
 94 | COURSE_FEATURE_START = USER_FEATURE_START + USER_FEATURE_COUNT
 95 | TOTAL_FEATURE_COUNT = COURSE_FEATURE_START + COURSE_FEATURE_COUNT
 96 | 
 97 | 
```

**Dòng 98–108.** `OBJECT_ACTIONS`: action → nhóm, **bỏ nhóm web_page**, nên hyperedge Object chỉ từ video/assignment/forum. `EDGE_FAMILIES`: thứ tự = mã family 0–3.

```python
 98 | # Used by 4_hypergraph.py and 6_hgnn.py.
 99 | OBJECT_ACTIONS = {}
100 | for group_name in ACTION_GROUPS:
101 |     if group_name == "web_page":
102 |         continue
103 |     for action in ACTION_GROUPS[group_name]:
104 |         OBJECT_ACTIONS[action] = group_name
105 | # Hyperedge families. Every node also gets one self-loop hyperedge when the graph is loaded.
106 | EDGE_FAMILIES = ("course", "object", "user", "self_loop")
107 | 
108 | 
```

**Dòng 109–115.** Hàm tạo một kịch bản: các family dùng, bộ cột, số lớp HGNN, có MLP không, có học W không.

```python
109 | # Used by 9_train.py and 10_summary.py
110 | def scenario(description, families=EDGE_FAMILIES, features="full", hgnn_layers=2, use_mlp=True,
111 |              learn_w=True):
112 |     return {"description": description, "families": tuple(families), "features": features,
113 |             "hgnn_layers": hgnn_layers, "use_mlp": use_mlp, "learn_w": learn_w}
114 | 
115 | 
```

**Dòng 116–134.** Danh sách kịch bản. `SCENARIOS_ALL` là các kịch bản chạy khi `--scenario all` (không gồm X1).

```python
116 | SCENARIOS = {
117 |     "M":  scenario("Mô hình chính: Course + Object + User + self-loop, full, HGNN 2 layer, có MLP"),
118 |     "A1": scenario("M bỏ Course", families=("object", "user", "self_loop")),
119 |     "A2": scenario("M bỏ Object", families=("course", "user", "self_loop")),
120 |     "A3": scenario("M bỏ User", families=("course", "object", "self_loop")),
121 |     "A4": scenario("M bỏ self-loop", families=("course", "object", "user")),
122 |     "W1": scenario("M không học W (W = I cố định, như HGNN gốc)", learn_w=False),
123 |     "F1": scenario("M chỉ dùng feature hành vi", features="feature"),
124 |     "F2": scenario("M dùng feature hành vi + người học", features="feature+user"),
125 |     "F3": scenario("M dùng feature hành vi + khóa học", features="feature+course"),
126 |     "L1": scenario("M với HGNN 1 layer", hgnn_layers=1),
127 |     "H":  scenario("M bỏ nhánh MLP, chỉ còn hypergraph", use_mlp=False),
128 |     # Optional, not part of "all": no neighbours at all (G = I, the HGNN branch becomes an MLP).
129 |     "X1": scenario("Tùy chọn: chỉ self-loop, không có hàng xóm", families=("self_loop",)),
130 | }
131 | 
132 | SCENARIOS_ALL = ("M", "A1", "A2", "A3", "A4", "W1", "F1", "F2", "F3", "L1", "H")
133 | 
134 | 
```

**Dòng 135–146.** Ngưỡng 0.5 và siêu tham số train. Lưu ý `patience` = 40 **lần chấm val** (mỗi 5 epoch), tức 200 epoch.

```python
135 | # Used by 9_train.py.
136 | THRESHOLD = 0.5  # p >= 0.5 -> predicted dropout
137 | TRAIN = {
138 |     "hidden_dim": 128,        
139 |     "dropout": 0.5,            
140 |     "learning_rate": 1e-3,     
141 |     "weight_decay": 5e-4,   
142 |     "family_weight_lr": 0.05,  
143 |     "epochs": 1000,         
144 |     "eval_every": 5,            
145 |     "patience": 40,           
146 | }
```

---

## 2. `1_download.py` — Tải dữ liệu

| Dòng | Làm gì |
|---|---|
| 11–13 | Tạo `data/raw/xuetangx` nếu chưa có |
| 15–22 | Với mỗi file trong `DOWNLOAD_FILES`: đã có thì bỏ qua, chưa có thì tải bằng `urlretrieve` |
| 26–27 | `download_oulad`: **hàm rỗng (`pass`)**, không ai gọi. OULAD có pipeline riêng trong `baseline/new_dataset/OULAD` |
| 30–34 | Dòng lệnh: `--raw-dir` |

**Đầu ra:** `prediction_data.tar.gz` (313 MB), `user_info.csv` (117 MB), `course_info.csv` (0.5 MB).

### Mã nguồn có chú giải

File `src/1_download.py`: 34 dòng, md5 `404ba6ea6fcf16685e1896748db1b921`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–9.** Tiêu đề, import, nạp `0_config` bằng `import_module` (vì tên file bắt đầu bằng số nên không `import` thường được).

```python
 1 | # 1. Download the raw XuetangX files (prediction_data, user_info, course_info).
 2 | # - XuetangX dataset http://moocdata.cn/data/user-activity
 3 | 
 4 | import argparse
 5 | import urllib.request
 6 | from importlib import import_module
 7 | from pathlib import Path
 8 | config = import_module("0_config")
 9 | 
```

**Dòng 10–23.** Tạo thư mục raw; với mỗi file: đã có thì bỏ qua, chưa có thì tải.

```python
10 | # Download XuetangX dataset
11 | def download_xuetangx(raw_dir=config.RAW):
12 |     raw_dir = Path(raw_dir)
13 |     raw_dir.mkdir(parents=True, exist_ok=True)
14 | 
15 |     for file_name, download_url in config.DOWNLOAD_FILES.items():
16 |         destination_path = raw_dir / file_name
17 |         if destination_path.is_file():
18 |             print(f"Already exists, skipping {file_name}", flush=True)
19 |             continue
20 | 
21 |         print(f"Downloading {file_name}", flush=True)
22 |         urllib.request.urlretrieve(download_url, destination_path)
23 | 
```

**Dòng 24–28.** Hàm rỗng, không ai gọi (OULAD có pipeline riêng).

```python
24 | 
25 | # Download OULAD dataset
26 | def download_oulad(raw_dir=config.RAW):
27 |     pass
28 | 
```

**Dòng 29–34.** Dòng lệnh: `--raw-dir` (mặc định `data/raw/xuetangx`).

```python
29 | 
30 | if __name__ == "__main__":
31 |     parser = argparse.ArgumentParser(description="Download the raw XuetangX dataset.")
32 |     parser.add_argument("--raw-dir", type=Path, default=config.RAW)
33 |     arguments = parser.parse_args()
34 |     download_xuetangx(arguments.raw_dir)
```

---

## 3. `2_preprocess.py` — Lọc sự kiện và chia split

**Đầu vào:** 3 file raw. **Đầu ra:** `simple/{train,validation,test}.csv`, mỗi dòng là **một sự kiện** với 14 cột:

`node_id, enroll_id, user_id, course_id, label, gender, education, birth, course_start, course_end, category, action, object_id, course_day`

| Dòng | Hàm / đoạn | Làm gì |
|---|---|---|
| 19–23 | `read_csv` | Đọc CSV UTF-8 thành từng dict, không nạp cả file vào RAM |
| 27–31 | `write_csv` | Ghi CSV từ tên cột + danh sách dòng |
| 35–45 | `read_prediction_data` | Đọc **tuần tự** (`r|gz`) các file trong tar.gz, chỉ lấy file có tên trong danh sách. Mỗi lần gọi là một lần đọc lại cả tar từ đầu |
| 49–60 | `split_train_enrollments` | Xáo danh sách enrollment train bằng `random.Random(1)`, n_train = ⌊0.8·n⌋; phần đầu → "train", phần sau → "validation". **Đầu vào phải là danh sách đã sort** (dòng 210) để split không đổi |
| 64–171 | `stream_events` | Ghi sự kiện của từng split (chi tiết bên dưới) |
| 65–67 | | `node_ids[split]`: enroll_id → node_id (0, 1, 2, … trong từng split) |
| 69 | | `enrollment_metadata`: enroll_id → (user_id, course_id), lấy từ dòng log **đầu tiên**, kể cả dòng bị loại sau đó |
| 77–82 | | Mở 3 file output, ghi dòng tiêu đề |
| 84–91 | | Với mỗi dòng log: bỏ enrollment không có nhãn; ghi nhớ metadata |
| 93–95 | | **Lọc 1:** bỏ action không thuộc 22 action |
| 97–100 | | **Lọc 2:** d = ngày(sự kiện) − ngày(mở khóa theo `course_info.start`); chỉ giữ 0 ≤ d < 35 |
| 101–104 | | Gán node_id theo **thứ tự xuất hiện sự kiện được giữ đầu tiên** trong tar |
| 106–123 | | Ghi dòng: ghép thông tin người học (`user_info`) và khóa học (`course_info`) |
| 125–157 | | Enrollment **không còn sự kiện nào** sau khi lọc (3,305 cái): vẫn ghi 1 dòng với action, object, course_day rỗng, node_id nối tiếp ở cuối (theo thứ tự enroll_id) |
| 162–171 | | Kiểm tra: tổng số node ghi ra = số enrollment có nhãn, nếu không thì báo lỗi |
| 175–225 | `preprocess` | Hàm chính |
| 182–188 | | **Đủ 3 file rồi thì BỎ QUA**. Muốn tạo lại phải xóa 3 file CSV trước |
| 192–198 | | Nạp `user_info` (theo user_id) và `course_info` (theo course_id) vào RAM |
| 200–208 | | Đọc nhãn train và test chính thức từ tar |
| 210–212 | | Chia train chính thức → train/val; test chính thức → test |
| 217–224 | | Gọi `stream_events` |

**Khi sửa cần nhớ:**
- Đổi cửa sổ ngày → sửa `OBSERVATION_DAYS` (config dòng 30), xóa 3 CSV, chạy lại 2 → 3 → 4.
- Đổi tỉ lệ hoặc seed của split → config dòng 28–29, cũng phải xóa CSV.
- `baseline/prepare_raw.py` import **chính hàm** `split_train_enrollments` và `read_prediction_data` của file này để baseline có cùng split với M.

### Mã nguồn có chú giải

File `src/2_preprocess.py`: 233 dòng, md5 `26e1f7392043932f5630736923f4bd8f`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–17.** Tiêu đề, nguồn tham khảo, import. `config_constant` = `0_config`.

```python
  1 | # 2. Split XuetangX into train / validation / test files
  2 | # - MST-GCN https://github.com/wudongze9/MST-GCN
  3 | # - CA-TFHN https://github.com/codeds27/CA-TFHN
  4 | # - CFIN https://github.com/wzfhaha/dropout_prediction
  5 | 
  6 | import argparse
  7 | import codecs
  8 | import csv
  9 | import random
 10 | import tarfile
 11 | from datetime import date
 12 | from importlib import import_module
 13 | from pathlib import Path
 14 | 
 15 | config_constant = import_module("0_config")
 16 | 
 17 | 
```

**Dòng 18–24.** Đọc CSV thành từng dict một (generator), không nạp cả file vào RAM.

```python
 18 | # Read a UTF-8 CSV file row by row, one dict per row
 19 | def read_csv(path):
 20 |     with open(path, newline="", encoding="utf-8") as source:
 21 |         reader = csv.DictReader(source)
 22 |         for row in reader:
 23 |             yield row
 24 | 
```

**Dòng 25–33.** Ghi CSV: dòng tiêu đề + các dòng dữ liệu.

```python
 25 | 
 26 | # Write a UTF-8 CSV file from column names and rows
 27 | def write_csv(path, columns, rows):
 28 |     with open(path, "w", newline="", encoding="utf-8") as output:
 29 |         writer = csv.writer(output)
 30 |         writer.writerow(columns)
 31 |         writer.writerows(rows)
 32 | 
 33 | 
```

**Dòng 34–47.** Đọc tuần tự (`r|gz`) các file trong tar.gz, chỉ những file có tên trong `selected_names`; trả về (tên file, dòng). Mỗi lần gọi là một lần đọc lại tar từ đầu.

```python
 34 | # Read selected CSV files inside prediction_data.tar.gz
 35 | def read_prediction_data(prediction_data_path, selected_names):
 36 |     with tarfile.open(prediction_data_path, "r|gz") as prediction_data:
 37 |         for member in prediction_data:
 38 |             name = Path(member.name).name
 39 |             if not member.isfile() or name not in selected_names:
 40 |                 continue
 41 | 
 42 |             with prediction_data.extractfile(member) as binary:
 43 |                 text = codecs.getreader("utf-8")(binary)                 # bytes -> text
 44 |                 for row in csv.DictReader(text):
 45 |                     yield name, row
 46 | 
 47 | 
```

**Dòng 48–62.** Xáo danh sách enrollment train bằng `random.Random(1)`, 80% đầu → train, 20% sau → validation. Đầu vào phải là danh sách đã sort (dòng 210), nếu không split sẽ khác.

```python
 48 | # Split enrollments into enrollment id -> "train" (80%) or "validation" (20%).
 49 | def split_train_enrollments(enrollment_ids):
 50 |     shuffled_ids = list(enrollment_ids)
 51 |     random.Random(config_constant.SPLIT_SEED).shuffle(shuffled_ids)            # random order, seed 1
 52 |     train_count = int(len(shuffled_ids) * config_constant.TRAIN_RATIO)         # n_train = ⌊0.8·n⌋
 53 | 
 54 |     split_by_enrollment = {}
 55 |     for position, enrollment_id in enumerate(shuffled_ids):
 56 |         if position < train_count:
 57 |             split_by_enrollment[enrollment_id] = "train"
 58 |         else:
 59 |             split_by_enrollment[enrollment_id] = "validation"
 60 |     return split_by_enrollment
 61 | 
 62 | 
```

**Dòng 63–83.** Chuẩn bị: `node_ids[split]` (enroll_id → node_id trong split), `enrollment_metadata` (user, course của mỗi enrollment), 14 cột output, mở 3 file và ghi tiêu đề.

```python
 63 | # Write the events and labeled of enrollment to train.csv, validation.csv or test.csv.
 64 | def stream_events(prediction_data_path, output_dir, users, courses, labels, split_by_enrollment):
 65 |     node_ids = {}
 66 |     for split_name in config_constant.SPLITS:
 67 |         node_ids[split_name] = {}
 68 |     # (user, course) of every enrollment, also those with no event in days 0–34.
 69 |     enrollment_metadata = {}
 70 |     columns = (
 71 |         "node_id", "enroll_id", "user_id", "course_id", "label",
 72 |         "gender", "education", "birth", "course_start", "course_end",
 73 |         "category", "action", "object_id", "course_day",
 74 |     )
 75 | 
 76 |     # One output file and one CSV writer per split; the files are closed at the end.
 77 |     files = {}
 78 |     writers = {}
 79 |     for split_name in config_constant.SPLITS:
 80 |         files[split_name] = open(output_dir / f"{split_name}.csv", "w", newline="", encoding="utf-8")
 81 |         writers[split_name] = csv.writer(files[split_name])
 82 |         writers[split_name].writerow(columns)
 83 | 
```

**Dòng 84–92.** Vòng qua mọi dòng log: lấy enroll/user/course; bỏ enrollment không có nhãn; ghi nhớ (user, course) từ dòng log đầu tiên của enrollment.

```python
 84 |     for _, row in read_prediction_data(prediction_data_path, config_constant.LOG_FILES):
 85 |         enrollment_id = int(row["enroll_id"])
 86 |         user_id = int(row["username"])
 87 |         course_id = row["course_id"]
 88 |         if enrollment_id not in split_by_enrollment:
 89 |             continue
 90 |         if enrollment_id not in enrollment_metadata:
 91 |             enrollment_metadata[enrollment_id] = (user_id, course_id)
 92 | 
```

**Dòng 93–96.** **Lọc 1:** bỏ action không thuộc 22 action.

```python
 93 |         action = row["action"]
 94 |         if action not in config_constant.ACTIONS:
 95 |             continue
 96 | 
```

**Dòng 97–105.** **Lọc 2:** d = ngày(sự kiện) − ngày(mở khóa); chỉ giữ 0 ≤ d < 35. Enrollment được gán node_id khi có sự kiện được giữ đầu tiên.

```python
 97 |         event_date = date.fromisoformat(row["time"][:10])
 98 |         course_start = date.fromisoformat(courses[course_id]["start"][:10])
 99 |         course_day = (event_date - course_start).days                       # d = date(event) − date(course start)
100 |         if 0 <= course_day < config_constant.OBSERVATION_DAYS:              # keep 0 ≤ d < 35
101 |             split_name = split_by_enrollment[enrollment_id]
102 |             split_node_ids = node_ids[split_name]
103 |             if enrollment_id not in split_node_ids:
104 |                 split_node_ids[enrollment_id] = len(split_node_ids)
105 | 
```

**Dòng 106–124.** Ghi 1 dòng sự kiện, ghép thêm thông tin người học (gender, education, birth) và khóa học (start, end, category).

```python
106 |             user = users[user_id]
107 |             course = courses[course_id]
108 |             writers[split_name].writerow((
109 |                 split_node_ids[enrollment_id],
110 |                 enrollment_id,
111 |                 user_id,
112 |                 course_id,
113 |                 labels[enrollment_id],
114 |                 user["gender"],
115 |                 user["education"],
116 |                 user["birth"],
117 |                 course["start"],
118 |                 course["end"],
119 |                 course["category"],
120 |                 action,
121 |                 row["object"].strip(),
122 |                 course_day,
123 |             ))
124 | 
```

**Dòng 125–158.** Enrollment không còn sự kiện nào sau khi lọc: vẫn ghi 1 dòng với action/object/course_day rỗng, để không mất node (giữ đủ 67,699 test). Không có dòng log nào → báo lỗi.

```python
125 |     # Keep enrollments with no event -> 0 in event count
126 |     empty_behavior_count = 0
127 |     for enrollment_id in sorted(split_by_enrollment):
128 |         split_name = split_by_enrollment[enrollment_id]
129 |         split_node_ids = node_ids[split_name]
130 |         if enrollment_id in split_node_ids:
131 |             continue
132 |         if enrollment_id not in enrollment_metadata:
133 |             raise ValueError(
134 |                 f"Enrollment {enrollment_id} has a label but no raw log row"
135 |             )
136 | 
137 |         user_id, course_id = enrollment_metadata[enrollment_id]
138 |         split_node_ids[enrollment_id] = len(split_node_ids)
139 |         user = users[user_id]
140 |         course = courses[course_id]
141 |         writers[split_name].writerow((
142 |             split_node_ids[enrollment_id],
143 |             enrollment_id,
144 |             user_id,
145 |             course_id,
146 |             labels[enrollment_id],
147 |             user["gender"],
148 |             user["education"],
149 |             user["birth"],
150 |             course["start"],
151 |             course["end"],
152 |             course["category"],
153 |             "",
154 |             "",
155 |             "",
156 |         ))
157 |         empty_behavior_count += 1
158 | 
```

**Dòng 159–173.** Đóng file; kiểm tra số node ghi ra = số enrollment có nhãn; in số enrollment không có sự kiện.

```python
159 |     for split_name in config_constant.SPLITS:
160 |         files[split_name].close()
161 | 
162 |     written_node_count = 0
163 |     for split_name in config_constant.SPLITS:
164 |         written_node_count += len(node_ids[split_name])
165 |     if written_node_count != len(split_by_enrollment):
166 |         raise RuntimeError("Not every labeled enrollment was written")
167 |     print(
168 |         f"Wrote {written_node_count} enrollments; preserved "
169 |         f"{empty_behavior_count} with zero observed events",
170 |         flush=True,
171 |     )
172 | 
173 | 
```

**Dòng 174–189.** Hàm chính. **Nếu đã có đủ 3 CSV thì bỏ qua**, nên muốn tạo lại phải xóa CSV trước.

```python
174 | # Run step 2: raw files -> three split CSVs (skipped when they already exist)
175 | def preprocess(raw_dir=config_constant.RAW, output_dir=config_constant.PROCESSED):
176 |     raw_dir = Path(raw_dir)
177 |     output_dir = Path(output_dir)
178 |     prediction_data_path = raw_dir / "prediction_data.tar.gz"
179 |     user_path = raw_dir / "user_info.csv"
180 |     course_path = raw_dir / "course_info.csv"
181 | 
182 |     all_exist = True
183 |     for split_name in config_constant.SPLITS:
184 |         if not (output_dir / f"{split_name}.csv").is_file():
185 |             all_exist = False
186 |     if all_exist:
187 |         print(f"Skipped existing outputs in {output_dir}", flush=True)
188 |         return
189 | 
```

**Dòng 190–199.** Nạp `user_info` (khóa = user_id) và `course_info` (khóa = course_id) vào RAM.

```python
190 |     output_dir.mkdir(parents=True, exist_ok=True)
191 | 
192 |     users = {}
193 |     for row in read_csv(user_path):
194 |         users[int(row["user_id"])] = row
195 | 
196 |     courses = {}
197 |     for row in read_csv(course_path):
198 |         courses[row["course_id"]] = row
199 | 
```

**Dòng 200–209.** Đọc nhãn train và test chính thức từ tar.

```python
200 |     train_labels = {}
201 |     test_labels = {}
202 |     for truth_name, row in read_prediction_data(prediction_data_path, config_constant.TRUTH_FILES):
203 |         enrollment_id = int(row["enroll_id"])
204 |         label = int(row["truth"])
205 |         if truth_name == "train_truth.csv":
206 |             train_labels[enrollment_id] = label
207 |         else:
208 |             test_labels[enrollment_id] = label
209 | 
```

**Dòng 210–213.** Chia train chính thức → train/val (danh sách đã sort); test chính thức → test.

```python
210 |     split_by_enrollment = split_train_enrollments(sorted(train_labels))
211 |     for enrollment_id in test_labels:
212 |         split_by_enrollment[enrollment_id] = "test"
213 | 
```

**Dòng 214–227.** Gộp nhãn, gọi `stream_events`, in thông báo.

```python
214 |     labels = {}
215 |     labels.update(train_labels)
216 |     labels.update(test_labels)
217 |     stream_events(
218 |         prediction_data_path,
219 |         output_dir,
220 |         users,
221 |         courses,
222 |         labels,
223 |         split_by_enrollment,
224 |     )
225 |     print(f"Saved preprocessed data to {output_dir}", flush=True)
226 | 
227 | 
```

**Dòng 228–233.** Dòng lệnh: `--raw-dir`, `--output-dir`.

```python
228 | if __name__ == "__main__":
229 |     parser = argparse.ArgumentParser(description="Split XuetangX into train, validation and test CSVs.")
230 |     parser.add_argument("--raw-dir", type=Path, default=config_constant.RAW)
231 |     parser.add_argument("--output-dir", type=Path, default=config_constant.PROCESSED)
232 |     arguments = parser.parse_args()
233 |     preprocess(arguments.raw_dir, arguments.output_dir)
```

---

## 4. `3_features.py` — Ma trận X (89 cột)

**Đầu vào:** 3 file CSV của bước 2. **Đầu ra:** `simple/{split}/X.npy` (float32, hàng thứ i = node_id i của split) và `simple/feature_names.csv`.

### 4.1 Bố cục 89 cột

| Cột | Tên | Cách tính | Chuẩn hóa |
|---|---|---|---|
| 0–34 | `activity_day_0..34` | Số sự kiện trong ngày d | log(1+c) rồi z-score |
| 35 | `total_events` | Tổng sự kiện ngày 0–34 | log(1+c) rồi z-score |
| 36 | `distinct_objects` | Số object khác nhau (**mọi action** có object id, kể cả web_page) | log(1+c) rồi z-score |
| 37–58 | `action_<nhóm>_<k>` | Số lần mỗi action (thứ tự = `config.ACTIONS`) | log(1+c) rồi z-score |
| 59–61 | `gender_female, gender_male, gender_missing` | one-hot | 0/1 |
| 62–69 | `education_*` (7) + `education_missing` | one-hot | 0/1 |
| 70 | `age` | năm mở khóa − năm sinh; = 0 nếu thiếu hoặc ngoài [10, 70] | z-score |
| 71–88 | `category_*` (17) + `category_missing` | one-hot | 0/1 |

μ, σ của mọi z-score **chỉ tính trên train**, rồi áp chung cho cả 3 split.

### 4.2 Từng đoạn

| Dòng | Hàm / đoạn | Làm gì |
|---|---|---|
| 22–32 | `feature_columns(name)` | Trả về chỉ số cột cho F1–F3 và full: "feature" = 0–58, "feature+user" = 0–70, "feature+course" = 0–58 + 71–88, "full" = 0–88. Dùng ở `5_graph_data.apply_scenario` |
| 36–83 | `feature_metadata` | Danh sách (tên cột, nguồn) → `feature_names.csv`. **Thứ tự phải khớp bảng 4.1**; thêm cột mới thì sửa cả ở đây |
| 87–94 | `set_one_hot` | Giá trị rỗng → cột missing; có trong danh sách → cột tương ứng; **không có trong danh sách → báo lỗi** (phải thêm vào config) |
| 99–105 | `age_at_course_start` | a = năm(course_start) − năm sinh; thiếu hoặc ngoài [10, 70] → 0 |
| 109–202 | `build_split_features` | Đếm cho một split |
| 110–113 | | Map action → cột 37 + vị trí trong `config.ACTIONS` |
| 119–127 | | Đọc từng dòng; dòng **đầu tiên** của node được giữ làm thông tin context (gender, education, birth, category) |
| 129–137 | | Dòng có action: +1 vào cột ngày `course_day`, +1 vào cột 35, +1 vào cột của action; object id (nếu có) đưa vào tập object của node |
| 139–140 | | Cột 36 = số phần tử của tập object |
| 142–155 | | Tạo 3 mảng: hành vi (int32, N×59), người học (N×12), khóa học (N×18) |
| 158–200 | | Với mỗi node: chép hành vi; chuẩn hóa giá trị thiếu (dòng 162–170); one-hot gender (173), education (181), tuổi (189), category (194) |
| 206–265 | `build_features` | Hàm chính |
| 210–220 | | Gọi `build_split_features` cho 3 split |
| 224–227 | | x = log(1 + c); μ_train, σ_train theo cột; cột hằng (σ = 0) thì đặt σ = 1 |
| 230–234 | | μ, σ của tuổi trên train; in tỉ lệ thiếu tuổi (71.6%) |
| 238–254 | | Áp cùng μ, σ cho cả 3 split; ghép [hành vi \| người học \| khóa học]; lưu `X.npy` float32 |
| 256–264 | | Ghi `feature_names.csv` |

**Khi sửa cần nhớ:**
- Thêm hoặc bớt cột → sửa config dòng 75–95 (chỉ số) **và** `feature_metadata` **và** `feature_columns`. `baseline/ML/LR.py` và `SVM.py` đang kiểm tra cứng `FEATURE_COUNT = 89`.
- Tuổi = 0 khi thiếu, rồi mới z-score; **không có cột age_missing** (em đã chốt ngày 04/10).

### Mã nguồn có chú giải

File `src/3_features.py`: 272 dòng, md5 `259b10cf4072b25df838ab6123b6815b`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–17.** Tiêu đề, import, dùng lại `read_csv` / `write_csv` của file 2.

```python
  1 | # 3. Build the node feature matrix X of train / validation / test
  2 | # - MST-GCN https://github.com/wudongze9/MST-GCN
  3 | # - CA-TFHN https://github.com/codeds27/CA-TFHN
  4 | # - CFIN https://github.com/wzfhaha/dropout_prediction
  5 | 
  6 | 
  7 | import argparse
  8 | from importlib import import_module
  9 | from pathlib import Path
 10 | 
 11 | import numpy as np
 12 | 
 13 | config = import_module("0_config")
 14 | preprocess_module = import_module("2_preprocess")
 15 | read_csv = preprocess_module.read_csv
 16 | write_csv = preprocess_module.write_csv
 17 | 
```

**Dòng 18–34.** Chỉ số cột cho từng bộ feature: `feature` = 0–58, `feature+user` = 0–70, `feature+course` = 0–58 + 71–88, `full` = 0–88 (dùng cho kịch bản F1–F3).

```python
 18 | 
 19 | # ("feature", "feature+user", "feature+course", "full")
 20 | # [0, 59) behavior | [59, 71) user | [71, 89) course
 21 | # Use it for experiment
 22 | def feature_columns(name):
 23 |     behavior = np.arange(0, config.BEHAVIOR_FEATURE_COUNT)                       # 35 days + total + objects + 22 actions
 24 |     user = np.arange(config.USER_FEATURE_START, config.COURSE_FEATURE_START)     # gender + education + age
 25 |     course = np.arange(config.COURSE_FEATURE_START, config.TOTAL_FEATURE_COUNT)  # category
 26 |     feature_sets = {
 27 |         "feature": behavior,
 28 |         "feature+user": np.concatenate([behavior, user]),
 29 |         "feature+course": np.concatenate([behavior, course]),
 30 |         "full": np.concatenate([behavior, user, course]),
 31 |     }
 32 |     return feature_sets[name]
 33 | 
 34 | 
```

**Dòng 35–56.** Tên cột (phần 1): 35 cột ngày, `total_events`, `distinct_objects`, 22 cột action theo nhóm.

```python
 35 | # Set header for file
 36 | def feature_metadata():
 37 |     metadata = []
 38 | 
 39 |     # Daily-count columns: total valid events on each observed course day.
 40 |     for day_number in range(config.DAY_FEATURE_COUNT):
 41 |         metadata.append((
 42 |             f"activity_day_{day_number}",
 43 |             f"course_day={day_number}",
 44 |         ))
 45 | 
 46 |     # Total events and distinct objects over the 35 days (MST-GCN)
 47 |     metadata.append(("total_events", "events in course days 0-34 (MST-GCN total_actions)"))
 48 |     metadata.append(("distinct_objects", "distinct objects used in course days 0-34 (MST-GCN unique_objects)"))
 49 | 
 50 |     # Action-count columns: total occurrences of each action across all days.
 51 |     for family, actions in config.ACTION_GROUPS.items():
 52 |         action_number = 1
 53 |         for action_name in actions:
 54 |             metadata.append((f"action_{family}_{action_number}", action_name))
 55 |             action_number += 1
 56 | 
```

**Dòng 57–85.** Tên cột (phần 2): 3 cột giới tính, 8 cột học vấn, `age`, 18 cột category. Thứ tự phải khớp với cách ghép X ở dòng 247–250.

```python
 57 |     # User-context columns: one-hot gender
 58 |     for gender in config.GENDERS:
 59 |         metadata.append((f"gender_{gender}", f"gender={gender}"))
 60 |     metadata.append(("gender_missing", "gender is missing"))
 61 | 
 62 |     # User-context columns: one-hot education
 63 |     for education in config.EDUCATIONS:
 64 |         clean_education = education.lower().replace("'", "").replace(" ", "_")
 65 |         metadata.append((
 66 |             f"education_{clean_education}",
 67 |             f"education={education}",
 68 |         ))
 69 |     metadata.append(("education_missing", "education is missing"))
 70 | 
 71 |     # User-context column: age -> standardized (CFIN)
 72 |     metadata.append((
 73 |         "age",
 74 |         f"course start year - birth year, {config.AGE_MISSING} if missing or outside "
 75 |         f"[{config.AGE_MIN}, {config.AGE_MAX}] (CFIN)",
 76 |     ))
 77 | 
 78 |     # Course-context columns: one-hot category
 79 |     for category in config.CATEGORIES:
 80 |         clean_category = category.replace(" ", "_")
 81 |         metadata.append((f"category_{clean_category}", f"category={category}"))
 82 |     metadata.append(("category_missing", "category is missing"))
 83 |     return metadata
 84 | 
 85 | 
```

**Dòng 86–96.** One-hot: rỗng → cột missing; có trong danh sách → cột tương ứng; **không có trong danh sách → báo lỗi**.

```python
 86 | # Set the one-hot cell of a categorical value
 87 | def set_one_hot(feature_matrix, row_index, value, vocabulary, feature_start):
 88 |     if not value:
 89 |         value_index = len(vocabulary)                                    # missing column
 90 |     elif value in vocabulary:
 91 |         value_index = vocabulary.index(value)
 92 |     else:
 93 |         raise ValueError(f"'{value}' is not in the vocabulary {vocabulary}; add it to 0_config.py")
 94 |     feature_matrix[row_index, feature_start + value_index] = 1.0
 95 | 
 96 | 
```

**Dòng 97–107.** Tuổi = năm mở khóa − năm sinh; thiếu hoặc ngoài [10, 70] → 0.

```python
 97 | # Age of a user when course starts
 98 | # age = year(course start) − birth year;  a = 0 if birth is missing or a < 10 or a > 70 (CFIN)
 99 | def age_at_course_start(birth, course_start):
100 |     if birth.lower() in config.MISSING_VALUES:
101 |         return config.AGE_MISSING                                        # birth missing -> a = 0
102 |     age = int(course_start[:4]) - int(float(birth))                      # a = year(course start) − birth year
103 |     if age < config.AGE_MIN or age > config.AGE_MAX:
104 |         return config.AGE_MISSING                                        # a outside [10, 70] -> a = 0
105 |     return age
106 | 
107 | 
```

**Dòng 108–117.** Bắt đầu đếm cho một split: map action → cột (37 + vị trí); 3 dict lưu dòng đầu, bộ đếm hành vi, tập object của mỗi node.

```python
108 | # Count data for feature
109 | def build_split_features(data_path):
110 |     action_indices = {}
111 |     # Column of every action, after the 35 day columns and the 2 summary columns.
112 |     for offset, action in enumerate(config.ACTIONS):
113 |         action_indices[action] = config.ACTION_FEATURE_START + offset
114 | 
115 |     nodes = {}
116 |     behavior_by_node = {}
117 |     objects_by_node = {}
```

**Dòng 118–128.** Đọc từng dòng; gặp node mới thì lưu dòng đầu (dùng cho context) và tạo bộ đếm 59 số 0.

```python
118 |     # Count events per day and per action, and collect the objects, for every node.
119 |     for row in read_csv(data_path):
120 |         node_id = int(row["node_id"])
121 |         if node_id not in nodes:
122 |             nodes[node_id] = row
123 |             behavior_by_node[node_id] = np.zeros(
124 |                 config.BEHAVIOR_FEATURE_COUNT,
125 |                 dtype=np.int32,
126 |             )
127 |             objects_by_node[node_id] = set()
128 | 
```

**Dòng 129–141.** Dòng có action: +1 cột ngày, +1 cột tổng (35), +1 cột action; object id (của **mọi** action) đưa vào tập. Cột 36 = kích thước tập object.

```python
129 |         action = row["action"].strip()
130 |         if action:
131 |             course_day = int(row["course_day"])
132 |             behavior_by_node[node_id][course_day] += 1                  
133 |             behavior_by_node[node_id][config.TOTAL_EVENTS_INDEX] += 1   
134 |             behavior_by_node[node_id][action_indices[action]] += 1     
135 |             object_id = row["object_id"].strip()
136 |             if object_id.lower() not in config.MISSING_VALUES:
137 |                 objects_by_node[node_id].add(object_id)
138 | 
139 |     for node_id, objects in objects_by_node.items():
140 |         behavior_by_node[node_id][config.OBJECT_COUNT_INDEX] = len(objects)
141 | 
```

**Dòng 142–156.** Tạo 3 mảng: hành vi N×59 (int32), người học N×12, khóa học N×18.

```python
142 |     node_count = len(nodes)
143 |     behavior_features = np.zeros(
144 |         (node_count, config.BEHAVIOR_FEATURE_COUNT),
145 |         dtype=np.int32,
146 |     )
147 | 
148 |     user_context = np.zeros(
149 |         (node_count, config.USER_FEATURE_COUNT),
150 |         dtype=np.float32,
151 |     )
152 |     course_context = np.zeros(
153 |         (node_count, config.COURSE_FEATURE_COUNT),
154 |         dtype=np.float32,
155 |     )
156 | 
```

**Dòng 157–171.** Điền hành vi theo node_id (hàng i = node_id i); chuẩn hóa giá trị thiếu của gender/education/category thành chuỗi rỗng.

```python
157 |     # Set behavior counts and encode user/course context for every node.
158 |     for node_id in sorted(nodes):
159 |         node = nodes[node_id]
160 |         behavior_features[node_id] = behavior_by_node[node_id]
161 | 
162 |         gender = node["gender"].strip()
163 |         education = node["education"].strip()
164 |         category = node["category"].strip()
165 |         if gender.lower() in config.MISSING_VALUES:
166 |             gender = ""
167 |         if education.lower() in config.MISSING_VALUES:
168 |             education = ""
169 |         if category.lower() in config.MISSING_VALUES:
170 |             category = ""
171 | 
```

**Dòng 172–204.** One-hot gender, education, tuổi, category; trả về 3 mảng.

```python
172 |         # one-hot gender
173 |         set_one_hot(
174 |             user_context,
175 |             node_id,
176 |             gender,
177 |             config.GENDERS,
178 |             config.GENDER_FEATURE_START,
179 |         )
180 |         # one-hot education level
181 |         set_one_hot(
182 |             user_context,
183 |             node_id,
184 |             education,
185 |             config.EDUCATIONS,
186 |             config.EDUCATION_FEATURE_START,
187 |         )
188 |         # calculate age, missing -> 0
189 |         user_context[node_id, config.AGE_FEATURE_INDEX] = age_at_course_start(
190 |             node["birth"].strip(),
191 |             node["course_start"].strip(),
192 |         )
193 |         # one-hot course category
194 |         set_one_hot(
195 |             course_context,
196 |             node_id,
197 |             category,
198 |             config.CATEGORIES,
199 |             config.CATEGORY_FEATURE_START,
200 |         )
201 | 
202 |     return behavior_features, user_context, course_context
203 | 
204 | 
```

**Dòng 205–221.** Hàm chính: đếm cho 3 split.

```python
205 | # Run step 3 and save into train-> X.npy / test -> X.npy / validation -> X.npy
206 | def build_features(output_dir=config.PROCESSED):
207 |     output_dir = Path(output_dir)
208 |     feature_data = {}
209 | 
210 |     for split_name in config.SPLITS:
211 |         split_dir = output_dir / split_name
212 |         split_dir.mkdir(parents=True, exist_ok=True)
213 |         behavior, user, course = build_split_features(
214 |             output_dir / f"{split_name}.csv"
215 |         )
216 |         feature_data[split_name] = {
217 |             "behavior": behavior,
218 |             "user": user,
219 |             "course": course,
220 |         }
221 | 
```

**Dòng 222–235.** Thống kê **chỉ trên train**: μ, σ của log(1 + c) theo từng cột (cột hằng thì σ = 1); μ, σ của tuổi.

```python
222 |     # log-transformed and then standardized
223 |     # x = (log(1 + c) - μ_train) / σ_train
224 |     train_counts = np.log1p(feature_data["train"]["behavior"].astype(np.float64))   # log(1 + c)
225 |     count_mean = np.mean(train_counts, axis=0)                           # μ_train of log(1 + c), per column
226 |     count_std = np.std(train_counts, axis=0)                             # σ_train of log(1 + c), per column
227 |     count_std[count_std == 0] = 1.0                                     # constant column -> z = 0
228 | 
229 |     # age uses z = (a - μ_train) / σ_train (CFIN)
230 |     train_age = feature_data["train"]["user"][:, config.AGE_FEATURE_INDEX].astype(np.float64)
231 |     age_mean = float(np.mean(train_age))                                  # μ_train of a
232 |     age_std = float(np.std(train_age))                                    # σ_train of a
233 |     print(f"Age: train mean={age_mean:.3f}, std={age_std:.3f}, "
234 |           f"unknown={np.mean(train_age == config.AGE_MISSING):.1%}", flush=True)
235 | 
```

**Dòng 236–255.** Áp cùng μ, σ cho 3 split: hành vi → (log(1 + c) − μ)/σ; tuổi → (a − μ)/σ; ghép [hành vi | người học | khóa học]; lưu `X.npy` float32.

```python
236 |     feature_paths = {}
237 |     # Same train statistics for every split.
238 |     for split_name in config.SPLITS:
239 |         split_data = feature_data[split_name]
240 |         behavior = np.log1p(split_data["behavior"].astype(np.float64))   # log(1 + c)
241 |         scaled_behavior = (behavior - count_mean) / count_std             # final X[:, 0:59]: z-score
242 |         user_features = split_data["user"].copy()
243 |         user_features[:, config.AGE_FEATURE_INDEX] = (
244 |             (user_features[:, config.AGE_FEATURE_INDEX] - age_mean) / age_std   # final X[:, 70]: z-score
245 |         )
246 | 
247 |         node_features = np.concatenate(
248 |             (scaled_behavior, user_features, split_data["course"]),
249 |             axis=1,
250 |         ).astype(np.float32, copy=False)
251 |         feature_path = output_dir / split_name / "X.npy"
252 |         np.save(feature_path, node_features)
253 |         feature_paths[split_name] = feature_path
254 |         print(f"Saved {feature_path}", flush=True)
255 | 
```

**Dòng 256–267.** Ghi `feature_names.csv` (chỉ số, tên, nguồn).

```python
256 |     feature_rows = []
257 |     for feature_index, metadata in enumerate(feature_metadata()):
258 |         feature_name, feature_source = metadata
259 |         feature_rows.append((feature_index, feature_name, feature_source))
260 |     write_csv(
261 |         output_dir / "feature_names.csv",
262 |         ("feature_index", "feature_name", "source"),
263 |         feature_rows,
264 |     )
265 |     return feature_paths
266 | 
267 | 
```

**Dòng 268–272.** Dòng lệnh: `--output-dir`.

```python
268 | if __name__ == "__main__":
269 |     parser = argparse.ArgumentParser(description="Build train, validation, and test node features.")
270 |     parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
271 |     arguments = parser.parse_args()
272 |     build_features(arguments.output_dir)
```

---

## 5. `4_hypergraph.py` — Hypergraph H

**Đầu vào:** 3 file CSV của bước 2. **Đầu ra:** `simple/hypergraph.npz` gồm:

| Khóa | Kiểu | Nghĩa |
|---|---|---|
| `node_ids`, `edge_ids` | int64, dài 3,222,033 | Cặp (v, e) có h(v,e) = 1, dạng COO |
| `edge_family` | int64, dài E | Mã family của mỗi hyperedge (0 course, 1 object, 2 user) |
| `labels` | **float32**, dài N | Nhãn dropout của mọi node (cả val/test, nhưng train chỉ dùng nhãn train) |
| `split` | int64, dài N | 0 train, 1 val, 2 test |

| Dòng | Hàm / đoạn | Làm gì |
|---|---|---|
| 16–20 | Hằng | Tên file; mã family lấy từ `config.EDGE_FAMILIES` |
| 29–48 | `read_split` | Lấy dòng đầu của mỗi node (cần `user_id`, `course_id`, `label`); gom object của node **chỉ từ 16 action video/assignment/forum** có object id. Khóa object = `"course_id\|nhóm\|object_id"`, nên cùng một object id ở hai nhóm khác nhau thành hai hyperedge khác nhau. Kiểm tra node_id liên tục 0..n−1 |
| 52–62 | `course_hyperedges` | Mỗi khóa học = 1 hyperedge chứa mọi enrollment của khóa (cả 3 split) |
| 66–76 | `object_hyperedges` | Mỗi object = 1 hyperedge chứa mọi enrollment đã tương tác với nó |
| 80–90 | `user_hyperedges` | Mỗi người học = 1 hyperedge chứa **mọi** enrollment của họ, kể cả khóa mở sau (quy tắc "any") |
| 94–149 | `build_hypergraph` | Hàm chính |
| 99–108 | | **Id toàn cục:** train 0 → 126,353; val nối tiếp; test nối tiếp (global = offset + node_id cục bộ). Đây cũng là thứ tự hàng khi `5_graph_data` ghép X |
| 110 | | Thứ tự hyperedge: Course (sort theo course_id) → Object (sort theo khóa) → User (sort theo user_id) |
| 112–115 | | **Chỉ giữ hyperedge có ≥ 2 node.** Người học chỉ có 1 enrollment thì không có hyperedge User |
| 118–126 | | Trải thành danh sách cặp (node, edge) |
| 132–139 | | Lưu `.npz` nén |
| 142–149 | | In số node theo split, số hyperedge theo family, thời gian |

**Lưu ý:** H chứa node của **cả 3 split** (transductive). Feature của node val/test có tham gia lan truyền, nhưng **nhãn** của chúng không vào loss (xem `9_train` dòng 118).

### Mã nguồn có chú giải

File `src/4_hypergraph.py`: 157 dòng, md5 `928d7d920897fc9e8283beea2ad905e7`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–22.** Tiêu đề, import, tên file output, mã family (0 course, 1 object, 2 user, 3 self_loop).

```python
  1 | # 4. Build one hypergraph H
  2 | # - HGNN https://github.com/iMoonLab/HGNN
  3 | # - MST-GCN https://github.com/wudongze9/MST-GCN
  4 | # - SIG-Net https://github.com/Noverse0/SIG-Net
  5 | 
  6 | import argparse
  7 | import time
  8 | from importlib import import_module
  9 | from pathlib import Path
 10 | 
 11 | import numpy as np
 12 | 
 13 | config = import_module("0_config")
 14 | read_csv = import_module("2_preprocess").read_csv
 15 | 
 16 | HYPERGRAPH_FILE = "hypergraph.npz"
 17 | COURSE = config.EDGE_FAMILIES.index("course")
 18 | OBJECT = config.EDGE_FAMILIES.index("object")
 19 | USER = config.EDGE_FAMILIES.index("user")
 20 | SELF_LOOP = config.EDGE_FAMILIES.index("self_loop")
 21 | 
 22 | 
```

**Dòng 23–27.** In log có giờ.

```python
 23 | # Print a message with the time
 24 | def log(message):
 25 |     print(f"[{time.strftime('%H:%M:%S')}][hypergraph] {message}", flush=True)
 26 | 
 27 | 
```

**Dòng 28–50.** Đọc 1 split: lưu dòng đầu của mỗi node; gom object **chỉ từ action video/assignment/forum** có object id, khóa = `course|nhóm|object`. Kiểm tra node_id liên tục 0..n−1.

```python
 28 | # Read a train / validation / test CSV
 29 | def read_split(path):
 30 |     nodes = {}
 31 |     objects = {}
 32 |     for row in read_csv(path):
 33 |         node_id = int(row["node_id"])
 34 |         if node_id not in nodes:
 35 |             nodes[node_id] = row
 36 |         action = row["action"]
 37 |         object_id = row["object_id"].strip()
 38 |         if action in config.OBJECT_ACTIONS and object_id.lower() not in config.MISSING_VALUES:
 39 |             family = config.OBJECT_ACTIONS[action]                   # video / assignment / forum
 40 |             if node_id not in objects:
 41 |                 objects[node_id] = set()
 42 |             objects[node_id].add(f"{row['course_id']}|{family}|{object_id}")
 43 |     if sorted(nodes) != list(range(len(nodes))):
 44 |         raise ValueError(f"{path}: node ids are not 0..n-1")
 45 |     node_list = []
 46 |     for node_id in range(len(nodes)):
 47 |         node_list.append(nodes[node_id])
 48 |     return node_list, objects
 49 | 
 50 | 
```

**Dòng 51–64.** Hyperedge Course: mỗi khóa = 1 hyperedge chứa mọi enrollment của khóa (cả 3 split).

```python
 51 | # Course hyperedges
 52 | def course_hyperedges(nodes):
 53 |     members = {}
 54 |     for node_id in range(len(nodes)):
 55 |         course_id = nodes[node_id]["course_id"]
 56 |         if course_id not in members:
 57 |             members[course_id] = []
 58 |         members[course_id].append(node_id)
 59 |     hyperedges = []
 60 |     for course_id in sorted(members):
 61 |         hyperedges.append((COURSE, course_id, members[course_id]))
 62 |     return hyperedges
 63 | 
 64 | 
```

**Dòng 65–78.** Hyperedge Object: mỗi object = 1 hyperedge chứa mọi enrollment đã tương tác với nó.

```python
 65 | # Object hyperedges
 66 | def object_hyperedges(objects):
 67 |     members = {}
 68 |     for node_id in sorted(objects):
 69 |         for key in objects[node_id]:
 70 |             if key not in members:
 71 |                 members[key] = []
 72 |             members[key].append(node_id)
 73 |     hyperedges = []
 74 |     for key in sorted(members):
 75 |         hyperedges.append((OBJECT, key, members[key]))
 76 |     return hyperedges
 77 | 
 78 | 
```

**Dòng 79–92.** Hyperedge User: mỗi người học = 1 hyperedge chứa **mọi** enrollment của họ.

```python
 79 | # User hyperedges
 80 | def user_hyperedges(nodes):
 81 |     members = {}
 82 |     for node_id in range(len(nodes)):
 83 |         user_id = int(nodes[node_id]["user_id"])
 84 |         if user_id not in members:
 85 |             members[user_id] = []
 86 |         members[user_id].append(node_id)
 87 |     hyperedges = []
 88 |     for user_id in sorted(members):
 89 |         hyperedges.append((USER, f"user|{user_id}", members[user_id]))
 90 |     return hyperedges
 91 | 
 92 | 
```

**Dòng 93–109.** Hàm chính: đọc 3 split nối tiếp; **id toàn cục = offset + node_id** (train trước, rồi val, rồi test); ghi mã split của mỗi node.

```python
 93 | # Run step 4: build the hypergraph
 94 | def build_hypergraph(output_dir=config.PROCESSED):
 95 |     started_at = time.perf_counter()
 96 |     output_dir = Path(output_dir)
 97 |     nodes = []
 98 |     objects = {}
 99 |     split = []
100 |     for split_id in range(len(config.SPLITS)):
101 |         split_name = config.SPLITS[split_id]
102 |         log(f"reading {split_name}.csv")
103 |         split_nodes, split_objects = read_split(output_dir / f"{split_name}.csv")
104 |         offset = len(nodes)
105 |         for local_id in split_objects:
106 |             objects[offset + local_id] = split_objects[local_id]       # global id = offset + local id
107 |         nodes.extend(split_nodes)
108 |         split.extend([split_id] * len(split_nodes))
109 | 
```

**Dòng 110–116.** Gộp 3 loại hyperedge (Course → Object → User); **chỉ giữ hyperedge có ≥ 2 node**.

```python
110 |     all_hyperedges = course_hyperedges(nodes) + object_hyperedges(objects) + user_hyperedges(nodes)
111 |     # Keep |e| ≥ 2
112 |     hyperedges = []
113 |     for family, key, members in all_hyperedges:
114 |         if len(members) >= 2:
115 |             hyperedges.append((family, key, members))
116 | 
```

**Dòng 117–127.** Trải thành các cặp (node, edge) = vị trí các ô 1 của H; ghi family của mỗi hyperedge.

```python
117 |     # Sparse incidence matrix H 
118 |     node_ids = []
119 |     edge_ids = []
120 |     edge_family = []
121 |     for edge_id in range(len(hyperedges)):
122 |         family, key, members = hyperedges[edge_id]
123 |         edge_family.append(family)
124 |         for node_id in members:
125 |             node_ids.append(node_id)
126 |             edge_ids.append(edge_id)
127 | 
```

**Dòng 128–140.** Lấy nhãn mọi node; lưu `hypergraph.npz` (node_ids, edge_ids, edge_family, labels float32, split).

```python
128 |     labels = []
129 |     for node in nodes:
130 |         labels.append(int(node["label"]))
131 | 
132 |     np.savez_compressed(
133 |         output_dir / HYPERGRAPH_FILE,
134 |         node_ids=np.asarray(node_ids, dtype=np.int64),
135 |         edge_ids=np.asarray(edge_ids, dtype=np.int64),
136 |         edge_family=np.asarray(edge_family, dtype=np.int64),
137 |         labels=np.asarray(labels, dtype=np.float32),
138 |         split=np.asarray(split, dtype=np.int64),
139 |     )
140 | 
```

**Dòng 141–151.** In số node mỗi split, số hyperedge mỗi family, số membership, thời gian.

```python
141 |     # Number of nodes of each split and hyperedges of each family, for the log.
142 |     split_counts = {}
143 |     for split_id in range(len(config.SPLITS)):
144 |         split_counts[config.SPLITS[split_id]] = split.count(split_id)
145 |     counts = {"course": 0, "object": 0, "user": 0}
146 |     for family, key, members in hyperedges:
147 |         counts[config.EDGE_FAMILIES[family]] += 1
148 |     log(f"saved {output_dir / HYPERGRAPH_FILE}: nodes={len(nodes):,} {split_counts}, hyperedges={counts}, "
149 |         f"memberships={len(node_ids):,}, elapsed={time.perf_counter() - started_at:.0f}s")
150 | 
151 | 
```

**Dòng 152–157.** Dòng lệnh: `--output-dir`.

```python
152 | if __name__ == "__main__":
153 |     parser = argparse.ArgumentParser(description="Build one hypergraph H (Course, Object, User) over "
154 |                                                  "train, validation and test.")
155 |     parser.add_argument("--output-dir", type=Path, default=config.PROCESSED)
156 |     arguments = parser.parse_args()
157 |     build_hypergraph(arguments.output_dir)
```

---

## 6. `5_graph_data.py` — Nạp dữ liệu và áp kịch bản

| Dòng | Hàm / đoạn | Làm gì |
|---|---|---|
| 16–36 | `load_graph` | Đọc `hypergraph.npz`; đọc 3 `X.npy`, kiểm tra số dòng mỗi file = số node của split đó (sai thì báo phải chạy lại bước 4); ghép X = [X_train; X_val; X_test] theo đúng thứ tự id toàn cục |
| 40–49 | `add_self_loops` | Thêm N hyperedge self-loop: node v thuộc hyperedge id = (số hyperedge hiện có + v), family = 3 |
| 53–95 | `apply_scenario` | Cắt dữ liệu theo kịch bản |
| 57–60 | | Danh sách family được giữ (không tính self_loop) |
| 63 | | Mặt nạ hyperedge được giữ |
| 65–73 | | Đánh số lại id hyperedge được giữ thành 0, 1, 2, … (hyperedge bị bỏ → −1) |
| 75–83 | | Chỉ giữ các cặp (node, edge) của hyperedge được giữ |
| 84–85 | | Kịch bản có "self_loop" thì gọi `add_self_loops` |
| 87 | | Chỉ lấy các cột X theo `features` (F1–F3) |
| 88–95 | | Trả về: features, labels, graph, và chỉ số node của train / val / test |

`load_graph` chỉ chạy **một lần**; `apply_scenario` chạy lại cho mỗi kịch bản (`9_train` dòng 249, 263).

### Mã nguồn có chú giải

File `src/5_graph_data.py`: 95 dòng, md5 `251c23b5b227d18bf9c9902b0ef6bce0`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–14.** Tiêu đề, import, dùng lại tên file và mã self-loop của file 4, `feature_columns` của file 3.

```python
 1 | # 5. Load the graph data
 2 | 
 3 | from importlib import import_module
 4 | from pathlib import Path
 5 | 
 6 | import numpy as np
 7 | 
 8 | config = import_module("0_config")
 9 | hypergraph_module = import_module("4_hypergraph")
10 | feature_columns = import_module("3_features").feature_columns
11 | HYPERGRAPH_FILE = hypergraph_module.HYPERGRAPH_FILE
12 | SELF_LOOP = hypergraph_module.SELF_LOOP
13 | 
14 | 
```

**Dòng 15–27.** Đọc `hypergraph.npz`: nhãn, split, các cặp (node, edge), family.

```python
15 | # Load the hypergraph H
16 | def load_graph(output_dir=config.PROCESSED):
17 |     output_dir = Path(output_dir)
18 |     with np.load(output_dir / HYPERGRAPH_FILE) as bundle:
19 |         labels = bundle["labels"]
20 |         split = bundle["split"]
21 |         graph = {
22 |             "num_nodes": len(labels),
23 |             "node_ids": bundle["node_ids"],
24 |             "edge_ids": bundle["edge_ids"],
25 |             "edge_family": bundle["edge_family"],
26 |         }
27 | 
```

**Dòng 28–38.** Đọc 3 `X.npy`, kiểm tra số dòng khớp số node mỗi split, ghép X = [train; val; test] (đúng thứ tự id toàn cục).

```python
28 |     split_features = []
29 |     for split_id in range(len(config.SPLITS)):
30 |         split_name = config.SPLITS[split_id]
31 |         features = np.load(output_dir / split_name / "X.npy")
32 |         if len(features) != np.sum(split == split_id):
33 |             raise ValueError(f"{split_name}/X.npy and hypergraph.npz do not align; rerun 4_hypergraph.py")
34 |         split_features.append(features)
35 |     features = np.concatenate(split_features)                       # X = [X_train; X_validation; X_test]
36 |     return {"features": features, "labels": labels, "split": split, "graph": graph}
37 | 
38 | 
```

**Dòng 39–51.** Thêm N self-loop: node v thuộc hyperedge mới có id = (số hyperedge hiện có + v), family 3.

```python
39 | # Add one self-loop hyperedge
40 | def add_self_loops(graph):
41 |     nodes = np.arange(graph["num_nodes"], dtype=np.int64)          
42 |     first_self_loop = len(graph["edge_family"])                   
43 |     self_loop_family = np.full(len(nodes), SELF_LOOP, dtype=np.int64)
44 | 
45 |     new_graph = dict(graph)
46 |     new_graph["node_ids"] = np.concatenate([graph["node_ids"], nodes])
47 |     new_graph["edge_ids"] = np.concatenate([graph["edge_ids"], first_self_loop + nodes])
48 |     new_graph["edge_family"] = np.concatenate([graph["edge_family"], self_loop_family])
49 |     return new_graph
50 | 
51 | 
```

**Dòng 52–64.** Áp kịch bản: lấy cột feature; danh sách family được giữ (trừ self_loop); mặt nạ hyperedge được giữ.

```python
52 | # This function for scenario when experiment
53 | def apply_scenario(data, scenario):
54 |     families = scenario["families"]
55 |     columns = feature_columns(scenario["features"])
56 | 
57 |     kept_families = []
58 |     for name in families:
59 |         if name != "self_loop":
60 |             kept_families.append(config.EDGE_FAMILIES.index(name))
61 | 
62 |     graph = data["graph"]
63 |     keep_edge = np.isin(graph["edge_family"], kept_families)               
64 | 
```

**Dòng 65–74.** Đánh số lại các hyperedge được giữ 0, 1, 2, …; bị bỏ → −1.

```python
65 |     new_edge_list = []
66 |     next_id = 0
67 |     for keep in keep_edge.tolist():
68 |         if keep:
69 |             new_edge_list.append(next_id)
70 |             next_id += 1
71 |         else:
72 |             new_edge_list.append(-1)
73 |     new_edge_id = np.asarray(new_edge_list, dtype=np.int64)
74 | 
```

**Dòng 75–86.** Giữ các cặp (node, edge) của hyperedge được giữ, đổi sang id mới; thêm self-loop nếu kịch bản có.

```python
75 |     keep_membership = keep_edge[graph["edge_ids"]]                          
76 |     kept_node_ids = graph["node_ids"][keep_membership]
77 |     kept_edge_ids = graph["edge_ids"][keep_membership]
78 |     scenario_graph = {
79 |         "num_nodes": graph["num_nodes"],
80 |         "node_ids": kept_node_ids,
81 |         "edge_ids": new_edge_id[kept_edge_ids],
82 |         "edge_family": graph["edge_family"][keep_edge],
83 |     }
84 |     if "self_loop" in families:
85 |         scenario_graph = add_self_loops(scenario_graph)
86 | 
```

**Dòng 87–95.** Lấy các cột X của kịch bản; trả về feature, nhãn, đồ thị và chỉ số node train/val/test.

```python
87 |     features = data["features"][:, columns]
88 |     return {
89 |         "features": np.ascontiguousarray(features, dtype=np.float32),
90 |         "labels": data["labels"],
91 |         "graph": scenario_graph,
92 |         "train_index": np.flatnonzero(data["split"] == 0),                    # nodes of the loss
93 |         "validation_index": np.flatnonzero(data["split"] == 1),               # nodes of early stopping
94 |         "test_index": np.flatnonzero(data["split"] == 2),                     # nodes scored once at the end
95 |     }
```

---

## 7. `6_hgnn.py` — Nhánh HGNN

**Công thức một lớp** (HGNN, Feng et al. 2019, có trọng số family):

  G = D_v^{-1/2} · H · W · D_e^{-1} · H^T · D_v^{-1/2},  với W = diag(w_e), D_e = diag(δ(e)), D_v = diag(d(v)), d(v) = Σ_e h(v,e)·w_e

  Z1 = ReLU(G·(X·Θ1 + b1)),  Z_g = ReLU(G·(Dropout(Z1)·Θ2 + b2))

| Dòng | Hàm / đoạn | Làm gì |
|---|---|---|
| 15 | `INITIAL_FAMILY_LOGIT` | θ ban đầu = log(e − 1), nên softplus(θ) = 1: **ban đầu W = I** |
| 19–33 | `prepare_graph` | Tạo tensor sparse H (N×E), H^T (E×N), δ(e) = số node mỗi hyperedge (`bincount`), và mã family của từng hyperedge. Gọi 1 lần cho mỗi seed |
| 38–53 | `HGNNEncoder.__init__` | Θ1 (D→128); Θ2 (128→128) nếu 2 lớp; 4 tham số θ_f. `learn_w=True` → θ là tham số học được; `False` (kịch bản W1) → θ là hằng, W = I |
| 56–57 | `family_weights` | w_f = softplus(θ_f) > 0 |
| 59–66 | `propagate` | Tính G·x **không dựng G**: d(v) = H·w (61); d^{-1/2}, chặn dưới 1e-6 (62); gom về hyperedge Σ_{u∈e} d(u)^{-1/2}·x_u (63); nhân w_e/δ(e) (64–65); trải về node rồi nhân d(v)^{-1/2} (66) |
| 68–76 | `forward` | w_e = w_{f(e)} (69); lớp 1: X·Θ1 + b1 → lan truyền → ReLU (70–71); 1 lớp thì dừng (72–73); dropout chỉ khi train (74); lớp 2 tương tự (75–76) |

**Lưu ý:** linear (Θ) được áp **trước** lan truyền, như HGNN gốc. Không có dropout ở đầu vào.

### Mã nguồn có chú giải

File `src/6_hgnn.py`: 76 dòng, md5 `633620d23878690db2882b645eede345`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–17.** Tiêu đề, import. θ ban đầu = log(e − 1), nên w = softplus(θ) = 1 (**ban đầu W = I**).

```python
 1 | # 6. Graph branch: an HGNN encoder 
 2 | # - HGNN https://github.com/iMoonLab/HGNN
 3 | # - HGNN+ https://github.com/iMoonLab/DeepHypergraph
 4 | 
 5 | import math
 6 | from importlib import import_module
 7 | 
 8 | import torch
 9 | from torch import nn
10 | from torch.nn import functional as F
11 | 
12 | config = import_module("0_config")
13 | 
14 | 
15 | INITIAL_FAMILY_LOGIT = math.log(math.e - 1)
16 | 
17 | 
```

**Dòng 18–35.** Chuẩn bị tensor: H (N×E) và H^T dạng sparse, δ(e) = số node mỗi hyperedge, family mỗi hyperedge.

```python
18 | # Get graph from graph_data and apply scenarion for experiment
19 | def prepare_graph(graph, device):
20 |     node_ids = torch.as_tensor(graph["node_ids"], dtype=torch.int64, device=device)
21 |     edge_ids = torch.as_tensor(graph["edge_ids"], dtype=torch.int64, device=device)
22 |     edge_family = torch.as_tensor(graph["edge_family"], dtype=torch.int64, device=device)
23 |     num_nodes = int(graph["num_nodes"])
24 |     num_edges = len(edge_family)
25 |     ones = torch.ones(len(node_ids), device=device)                       
26 |     return {
27 |         "H": torch.sparse_coo_tensor(torch.stack([node_ids, edge_ids]), ones,
28 |                                      (num_nodes, num_edges), check_invariants=True).coalesce(),
29 |         "H_T": torch.sparse_coo_tensor(torch.stack([edge_ids, node_ids]), ones,
30 |                                        (num_edges, num_nodes), check_invariants=True).coalesce(),
31 |         "edge_size": torch.bincount(edge_ids, minlength=num_edges).float(),  
32 |         "edge_family": edge_family,
33 |     }
34 | 
35 | 
```

**Dòng 36–54.** Khởi tạo: Θ1 (D→128); Θ2 (128→128) nếu 2 lớp; 4 tham số θ_f (học được, hoặc hằng nếu W1).

```python
36 | # HGNN
37 | class HGNNEncoder(nn.Module):
38 |     def __init__(self, input_dim, hidden_dim, dropout, layers=2, learn_w=True):
39 |         super().__init__()
40 |         if layers not in (1, 2):
41 |             raise ValueError("layers must be 1 or 2")
42 |         self.layer1 = nn.Linear(input_dim, hidden_dim)                       # Θ1, b1
43 |         if layers == 2:
44 |             self.layer2 = nn.Linear(hidden_dim, hidden_dim)                  # Θ2, b2 (2 layers only)
45 |         else:
46 |             self.layer2 = None
47 |         self.dropout = dropout
48 |         self.learn_w = learn_w
49 |         logits = torch.full((len(config.EDGE_FAMILIES),), INITIAL_FAMILY_LOGIT)   # θ_f, one per hyperedge family
50 |         if learn_w:
51 |             self.family_logits = nn.Parameter(logits)                        # learned: W = diag(softplus(θ_f(e)))
52 |         else:
53 |             self.register_buffer("family_logits", logits)                    # fixed: softplus(θ) = 1, so W = I
54 | 
```

**Dòng 55–58.** w_f = softplus(θ_f) > 0.

```python
55 |     # Weight of every hyperedge family: w_f = softplus(θ_f) > 0.
56 |     def family_weights(self):
57 |         return F.softplus(self.family_logits)
58 | 
```

**Dòng 59–67.** Lan truyền G·x mà không dựng G: d(v) = H·w; chia d(v)^{1/2}; gom về hyperedge; nhân w_e/δ(e); trải về node; chia d(v)^{1/2}.

```python
59 |     def propagate(self, x, graph, w):
60 |         # x'_v = d(v)^-1/2 · Σ_e h(v,e) · (w_e/δ(e)) · Σ_{u∈e} d(u)^-1/2 · x_u
61 |         node_degree = torch.sparse.mm(graph["H"], w.unsqueeze(1))      # d(v) = Σ_e h(v,e)·w_e, [N, 1]
62 |         node_scale = node_degree.clamp_min(1e-6).rsqrt()                 # d(v)^-1/2
63 |         edge_x = torch.sparse.mm(graph["H_T"], x * node_scale)           # Σ_{u∈e} d(u)^-1/2·x_u
64 |         edge_scale = (w / graph["edge_size"]).unsqueeze(1)               # w_e/δ(e), [E] -> [E, 1]
65 |         edge_x = edge_x * edge_scale                                     # m_e = (w_e/δ(e))·Σ_{u∈e} d(u)^-1/2·x_u
66 |         return torch.sparse.mm(graph["H"], edge_x) * node_scale          # x'_v = d(v)^-1/2·Σ_e h(v,e)·m_e
67 | 
```

**Dòng 68–76.** Forward: w_e = w_{f(e)}; Z1 = ReLU(G·(X·Θ1 + b1)); 1 lớp thì trả Z1; 2 lớp: dropout (chỉ khi train), Z_g = ReLU(G·(Z1·Θ2 + b2)).

```python
68 |     def forward(self, x, graph):
69 |         w = self.family_weights()[graph["edge_family"]]                   # w_e = w_f(e)
70 |         h = self.layer1(x)                                               # X·Θ1 + b1
71 |         z = F.relu(self.propagate(h, graph, w))                          # Z1 = ReLU(G·(X·Θ1 + b1))
72 |         if self.layer2 is None:
73 |             return z                                                     # 1 layer: Z_g = Z1
74 |         z = F.dropout(z, self.dropout, self.training)                    # only while training
75 |         h = self.layer2(z)                                               # Z1·Θ2 + b2
76 |         return F.relu(self.propagate(h, graph, w))                       # Z_g = ReLU(G·(Z1·Θ2 + b2))
```

---

## 8. `7_mlp.py` — Nhánh MLP

| Dòng | Làm gì |
|---|---|
| 10–18 | z_s = ReLU(Dropout(ReLU(x·A1 + c1))·A2 + c2), kích thước D → 128 → 128 |
| 22–23 | `forward`: chỉ dùng feature của **chính node**, không dùng đồ thị |

### Mã nguồn có chú giải

File `src/7_mlp.py`: 23 dòng, md5 `2c734e9ccf8e4fcea864b30fc5f177a6`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–5.** Tiêu đề, import.

```python
 1 | # 7. MLP branch
 2 | # - UniGNN / UniGCNII https://github.com/OneForward/UniGNN
 3 | 
 4 | from torch import nn
 5 | 
```

**Dòng 6–19.** MLP 2 lớp: z_s = ReLU(Dropout(ReLU(x·A1 + c1))·A2 + c2), D → 128 → 128.

```python
 6 | # MLP branch: z_s from the node's own features.
 7 | class MLPEncoder(nn.Module):
 8 |     # Two layers A1, c1 and A2, c2:
 9 |     #   z_s = ReLU( Dropout(ReLU(x·A1 + c1))·A2 + c2 )
10 |     def __init__(self, input_dim, hidden_dim, dropout):
11 |         super().__init__()
12 |         self.layers = nn.Sequential(
13 |             nn.Linear(input_dim, hidden_dim),     # x·A1 + c1
14 |             nn.ReLU(),
15 |             nn.Dropout(dropout),                  # only while training
16 |             nn.Linear(hidden_dim, hidden_dim),    # ·A2 + c2
17 |             nn.ReLU(),
18 |         )
19 | 
```

**Dòng 20–23.** Forward: chỉ dùng feature của chính node.

```python
20 |     # Representation from the node's own features.
21 |     # Shapes: x [B, D] -> z_s [B, hidden].
22 |     def forward(self, x):
23 |         return self.layers(x)
```

---

## 9. `8_model.py` — Ghép hai nhánh

| Dòng | Làm gì |
|---|---|
| 17–26 | Tạo nhánh HGNN; có MLP thì classifier nhận 256 chiều [z_g ‖ z_s], không có MLP (kịch bản H) thì 128 chiều |
| 29–33 | `join`: nối z_g và z_s |
| 37–41 | `forward`: logit = [z_g ‖ z_s]·u + b cho **mọi N node**, trả về vector dài N |
| 44–54 | `weight_summary`: w_f = softplus(θ_f) của các family đang dùng (ghi vào results.csv); không học W hoặc family bị bỏ → để trống |

### Mã nguồn có chú giải

File `src/8_model.py`: 54 dòng, md5 `a4dceccee24a483b3c573fbe9cfbab18`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–13.** Tiêu đề, import, nạp `HGNNEncoder` và `MLPEncoder`.

```python
 1 | # 8. Dropout prediction model
 2 | 
 3 | from importlib import import_module
 4 | 
 5 | import torch
 6 | from torch import nn
 7 | from torch.nn import functional as F
 8 | 
 9 | config = import_module("0_config")
10 | HGNNEncoder = import_module("6_hgnn").HGNNEncoder
11 | MLPEncoder = import_module("7_mlp").MLPEncoder
12 | 
13 | 
```

**Dòng 14–27.** Khởi tạo: nhánh HGNN; có MLP thì classifier nhận 256 chiều [z_g ‖ z_s], không có thì 128.

```python
14 | # Full model: HGNN branch, optional MLP branch, linear classifier.
15 | class DropoutModel(nn.Module):
16 |     # Both branches and the classifier u, b; use_mlp=False drops the MLP branch (scenario H).
17 |     def __init__(self, input_dim, hidden_dim, dropout, hgnn_layers=2, use_mlp=True, learn_w=True):
18 |         super().__init__()
19 |         self.hgnn = HGNNEncoder(input_dim, hidden_dim, dropout, hgnn_layers, learn_w)   # graph branch
20 |         if use_mlp:
21 |             self.mlp = MLPEncoder(input_dim, hidden_dim, dropout)                   # own-feature branch
22 |             classifier_dim = 2 * hidden_dim                                          # [z_g ‖ z_s]
23 |         else:
24 |             self.mlp = None
25 |             classifier_dim = hidden_dim                                              # z_g
26 |         self.classifier = nn.Linear(classifier_dim, 1)                              # u, b
27 | 
```

**Dòng 28–34.** Nối z_g với z_s (bỏ qua nếu không có MLP).

```python
28 |     # [z_g ‖ z_s] with the MLP branch, z_g alone without it.
29 |     def join(self, z_graph, x):
30 |         if self.mlp is None:
31 |             return z_graph
32 |         z_self = self.mlp(x)
33 |         return torch.cat([z_graph, z_self], dim=1)
34 | 
```

**Dòng 35–42.** Forward: logit = [z_g ‖ z_s]·u + b cho mọi N node.

```python
35 |     # Logits of every node (train + validation + test) on H.
36 |     # Shapes: x [N, D] -> logits [N].
37 |     def forward(self, x, graph):
38 |         z_graph = self.hgnn(x, graph)
39 |         z = self.join(z_graph, x)
40 |         logits = self.classifier(z)                                     # logit = [z_g ‖ z_s]·u + b, [N, 1]
41 |         return logits.squeeze(-1)                                       # [N]
42 | 
```

**Dòng 43–54.** Trọng số family w_f để ghi kết quả; không học W hoặc family không dùng → để trống.

```python
43 |     # Learned family weights for the results table
44 |     def weight_summary(self, families):
45 |         with torch.no_grad():
46 |             weights = F.softplus(self.hgnn.family_logits).tolist()  # w_f = softplus(θ_f)
47 |         summary = {}
48 |         for index in range(len(config.EDGE_FAMILIES)):
49 |             name = config.EDGE_FAMILIES[index]
50 |             if self.hgnn.learn_w and name in families:
51 |                 summary[f"w_{name}"] = weights[index]
52 |             else:
53 |                 summary[f"w_{name}"] = ""                           # W not learned, or family not used
54 |         return summary
```

---

## 10. `9_train.py` — Huấn luyện và đánh giá

**Đầu vào:** `hypergraph.npz` + `X.npy`. **Đầu ra:** mỗi (kịch bản, seed) **nối thêm 1 dòng** vào `outputs/results.csv` (31 cột, dòng 21–30).

| Dòng | Hàm / đoạn | Làm gì |
|---|---|---|
| 21–30 | `RESULT_COLUMNS` | Tên cột của results.csv; phải khớp với khóa của `row` (dòng 187–207) và với `10_summary.py` |
| 39–43 | `set_seed` | Đặt seed cho random, numpy, torch, cuda |
| 47–52 | `resolve_device` | "auto" → cuda nếu có, không thì cpu |
| 57–71 | `metrics` | ŷ = 1[p ≥ 0.5]; AUC, AUPRC (average precision), Acc, P, R, F1 của lớp dropout (nhãn 1). `baseline/ML` và CFIN chép y nguyên hàm này |
| 75–82 | `predict` | eval mode (tắt dropout), không tính gradient, forward **toàn bộ** N node, lấy σ(logit) của các node cần |
| 86–102 | `append_result` | Nối 1 dòng vào results.csv; file mới thì ghi tiêu đề trước; số thực ghi 6 chữ số |
| 106–208 | `train_one_seed` | Huấn luyện một (kịch bản, seed) |
| 109 | | Đặt seed |
| 110–118 | | Đưa X, H, chỉ số split lên device. **Chỉ nhãn train lên device** (118); nhãn val/test ở numpy, chỉ dùng để chấm điểm |
| 120–123 | | Tạo `DropoutModel` |
| 124–136 | | Adam, 2 nhóm tham số: (Θ, A, u) với lr 1e-3, weight_decay 5e-4; (θ_f) với lr 0.05, **không** weight decay |
| 141–151 | | **Mỗi epoch = 1 bước full-batch:** forward cả đồ thị, loss BCE chỉ trên node train, cắt gradient ‖g‖ ≤ 5, cập nhật |
| 153–155 | | Mỗi 50 epoch in loss và train AUC |
| 158–172 | | **Early stopping:** cứ 5 epoch (`eval_every`) chấm val AUC một lần. Tốt hơn → lưu bản sao trọng số trong RAM, `stale` = 0; không tốt hơn → `stale` += 1. Dừng khi `stale` ≥ 40 (`patience`). **Patience tính theo số lần chấm**, nên 40 × 5 = **200 epoch** không cải thiện mới dừng |
| 175–184 | | Khôi phục trọng số tốt nhất; chấm val và test **một lần** |
| 187–208 | | Ghi dòng kết quả: cấu hình, epoch tốt nhất, số epoch đã chạy, chỉ số val/test, w_f, thời gian |
| 212–223 | `scenario_codes` | "all" → `SCENARIOS_ALL`; kiểm tra mã kịch bản hợp lệ |
| 226–246 | CLI | `--scenario`, `--seeds`, `--epochs`, `--eval-every`, `--patience`, `--device`, `--output-dir` |
| 249–254 | | Nạp dữ liệu **một lần** |
| 257–265 | | Với mỗi kịch bản: áp kịch bản, chạy đủ các seed |

### Mã nguồn có chú giải

File `src/9_train.py`: 265 dòng, md5 `d047f97d47096b3e471a46b0a8519a43`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–20.** Tiêu đề, import, nạp các file 0, 5, 6, 8.

```python
  1 | # 9. Run the experiment scenarios 
  2 | # - HGNN https://github.com/iMoonLab/HGNN
  3 | 
  4 | import argparse
  5 | import copy
  6 | import csv
  7 | import random
  8 | import time
  9 | from importlib import import_module
 10 | 
 11 | import numpy as np
 12 | import torch
 13 | from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
 14 | from torch.nn import functional as F
 15 | 
 16 | config = import_module("0_config")
 17 | graph_data = import_module("5_graph_data")
 18 | prepare_graph = import_module("6_hgnn").prepare_graph
 19 | DropoutModel = import_module("8_model").DropoutModel
 20 | 
```

**Dòng 21–32.** Tên 31 cột của `outputs/results.csv`; phải khớp với khóa của `row` (dòng 187–207) và với `10_summary.py`.

```python
 21 | RESULT_COLUMNS = (
 22 |     "time", "scenario", "seed",
 23 |     "families", "features", "hgnn_layers", "use_mlp", "learn_w",
 24 |     "epochs", "eval_every", "patience", "hidden_dim", "dropout", "learning_rate", "weight_decay",
 25 |     "best_epoch", "epochs_run",
 26 |     "val_auc", "val_auprc", "val_f1",
 27 |     "test_auc", "test_auprc", "test_accuracy", "test_precision", "test_recall", "test_f1",
 28 |     "w_course", "w_object", "w_user", "w_self_loop",
 29 |     "minutes",
 30 | )
 31 | 
 32 | 
```

**Dòng 33–37.** In log có giờ và phạm vi.

```python
 33 | # Print a message with the time and a scope, e.g. [train].
 34 | def log(scope, message):
 35 |     print(f"[{time.strftime('%H:%M:%S')}][{scope}] {message}", flush=True)
 36 | 
 37 | 
```

**Dòng 38–45.** Đặt seed cho random, numpy, torch, cuda.

```python
 38 | # Seed Python, NumPy and PyTorch.
 39 | def set_seed(seed):
 40 |     random.seed(seed)
 41 |     np.random.seed(seed)
 42 |     torch.manual_seed(seed)
 43 |     torch.cuda.manual_seed_all(seed)
 44 | 
 45 | 
```

**Dòng 46–54.** Chọn thiết bị: "auto" → cuda nếu có.

```python
 46 | # Torch device from "auto", "cpu" or "cuda" ("auto" = cuda when available).
 47 | def resolve_device(name):
 48 |     if name == "auto":
 49 |         if torch.cuda.is_available():
 50 |             return torch.device("cuda")
 51 |         return torch.device("cpu")
 52 |     return torch.device(name)
 53 | 
 54 | 
```

**Dòng 55–73.** Chỉ số của lớp dropout tại ngưỡng 0.5: AUC, AUPRC, Acc, P, R, F1.

```python
 55 | # Metrics of the dropout class (label 1) at threshold 0.5: auc, auprc, accuracy, precision,
 56 | # recall, f1.
 57 | def metrics(labels, probabilities):
 58 |     predicted = probabilities >= config.THRESHOLD                       # ŷ = 1[p ≥ 0.5]
 59 |     # PRECISION = TP/(TP + FP)
 60 |     # RRCALL = TP/(TP + FN)
 61 |     # F1 = 2·P·R/(P + R)
 62 |     precision, recall, f1, _ = precision_recall_fscore_support(
 63 |         labels, predicted, labels=[1], zero_division=0)
 64 |     return {
 65 |         "auc": float(roc_auc_score(labels, probabilities)),             # AUC = P(p_dropout > p_non-dropout)
 66 |         "auprc": float(average_precision_score(labels, probabilities)), # AUPRC = Σ_n (R_n − R_{n−1})·P_n
 67 |         "accuracy": float(np.mean(predicted == labels)),                # Acc = (TP + TN)/n
 68 |         "precision": float(precision[0]),
 69 |         "recall": float(recall[0]),
 70 |         "f1": float(f1[0]),
 71 |     }
 72 | 
 73 | 
```

**Dòng 74–84.** Dự đoán: tắt dropout, không tính gradient, forward toàn bộ N node, lấy σ(logit) của các node cần.

```python
 74 | # Dropout probabilities
 75 | def predict(model, x, graph, index):
 76 |     was_training = model.training
 77 |     model.eval()                                                         # no dropout
 78 |     with torch.no_grad():                                                # no gradients while scoring
 79 |         logits = model(x, graph)                                         # every node of H
 80 |         probabilities = torch.sigmoid(logits[index])                     # p = σ(logit) of the chosen nodes
 81 |     model.train(was_training)
 82 |     return probabilities.cpu().numpy()
 83 | 
 84 | 
```

**Dòng 85–104.** Nối 1 dòng vào results.csv (file mới thì ghi tiêu đề trước), số thực 6 chữ số.

```python
 85 | # Append one row to outputs/results.csv (header written when the file is new).
 86 | def append_result(row):
 87 |     config.OUTPUTS.mkdir(parents=True, exist_ok=True)
 88 |     is_new = not config.RESULTS_CSV.exists()
 89 |     with open(config.RESULTS_CSV, "a", newline="", encoding="utf-8") as output:
 90 |         writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
 91 |         if is_new:
 92 |             writer.writeheader()
 93 |         # Floats are written with 6 decimals, everything else as it is.
 94 |         text_row = {}
 95 |         for name in row:
 96 |             value = row[name]
 97 |             if isinstance(value, float):
 98 |                 text_row[name] = f"{value:.6f}"
 99 |             else:
100 |                 text_row[name] = value
101 |         writer.writerow(text_row)
102 |     log("result", f"appended to {config.RESULTS_CSV}")
103 | 
104 | 
```

**Dòng 105–119.** Một (kịch bản, seed): đặt seed; đưa X, H, chỉ số split lên thiết bị; **chỉ nhãn train lên thiết bị**.

```python
105 | # Train
106 | def train_one_seed(code, seed, settings, data, device):
107 |     started_at = time.perf_counter()
108 |     scenario = config.SCENARIOS[code]
109 |     set_seed(seed)
110 |     x = torch.as_tensor(data["features"], device=device)                     # X of every node of H
111 |     graph = prepare_graph(data["graph"], device)
112 |     index = {}
113 |     split_labels = {}
114 |     for split in ("train", "validation", "test"):
115 |         index[split] = torch.as_tensor(data[f"{split}_index"], device=device)
116 |         split_labels[split] = data["labels"][data[f"{split}_index"]]       # NumPy, read by metrics
117 |     # Only the train labels go to the device: the loss sees no validation or test label.
118 |     train_labels = torch.as_tensor(split_labels["train"], device=device)
119 | 
```

**Dòng 120–137.** Tạo mô hình; Adam với 2 nhóm: (Θ, A, u) lr 1e-3, L2 5e-4; (θ_f) lr 0.05, không L2.

```python
120 |     model = DropoutModel(x.shape[1], settings["hidden_dim"], settings["dropout"],
121 |                          hgnn_layers=scenario["hgnn_layers"], use_mlp=scenario["use_mlp"],
122 |                          learn_w=scenario["learn_w"])
123 |     model = model.to(device)
124 |     family_logits = model.hgnn.family_logits
125 | 
126 |     # Every weight except the family weights W: Θ, A, u.
127 |     other_parameters = []
128 |     for parameter in model.parameters():
129 |         if parameter is not family_logits:
130 |             other_parameters.append(parameter)
131 | 
132 |     # Adam with L2 (weight_decay λ): g = ∇L + λ·θ, then θ ← θ − lr·m̂/(√v̂ + ε)
133 |     groups = [{"params": other_parameters}]                                          # Θ, A, u: lr, L2
134 |     if scenario["learn_w"]:                                                          # W: own lr, no L2
135 |         groups.append({"params": [family_logits], "lr": settings["family_weight_lr"], "weight_decay": 0.0})
136 |     optimizer = torch.optim.Adam(groups, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
137 | 
```

**Dòng 138–152.** Khởi tạo `best` (val AUC tốt nhất, epoch, trọng số) và `stale`. Mỗi epoch = 1 bước full-batch: forward cả đồ thị, BCE trên node train, cắt gradient ‖g‖ ≤ 5, cập nhật.

```python
138 |     best = {"auc": -1.0, "epoch": 0, "state": None}
139 |     stale = 0          # validations in a row without a better AUC
140 |     epoch = 0
141 |     for epoch in range(1, settings["epochs"] + 1):
142 |         # One full-batch training step: forward over the whole H, loss on the train nodes only.
143 |         model.train()
144 |         optimizer.zero_grad()
145 |         logits = model(x, graph)                                         # every node of H
146 |         train_logits = logits[index["train"]]
147 |         # L = −(1/|V_train|)·Σ_{v∈V_train} [y·log p + (1−y)·log(1−p)]
148 |         loss = F.binary_cross_entropy_with_logits(train_logits, train_labels)
149 |         loss.backward()
150 |         torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)          # ‖g‖ ≤ 5
151 |         optimizer.step()
152 | 
```

**Dòng 153–156.** Mỗi 50 epoch in loss và train AUC.

```python
153 |         if epoch % 50 == 0:
154 |             train_auc = roc_auc_score(split_labels["train"], train_logits.detach().cpu().numpy())
155 |             log("train", f"{code} seed {seed} epoch {epoch}: loss={loss.item():.4f}, train_auc={train_auc:.4f}")
156 | 
```

**Dòng 157–173.** Early stopping: mỗi 5 epoch chấm val AUC; tốt hơn → lưu trọng số (trong RAM), `stale` = 0; không → `stale` + 1; `stale` ≥ 40 thì dừng (= 200 epoch).

```python
157 |         # Early stopping: validation AUC every eval_every epochs, keep the best weights in memory.
158 |         if epoch % settings["eval_every"] == 0:
159 |             validation_probabilities = predict(model, x, graph, index["validation"])
160 |             validation_auc = roc_auc_score(split_labels["validation"], validation_probabilities)
161 |             if validation_auc > best["auc"]:                             # best = argmax_epoch AUC_val
162 |                 best = {"auc": validation_auc, "epoch": epoch,
163 |                         "state": copy.deepcopy(model.state_dict())}      # kept in memory, no file
164 |                 stale = 0
165 |             else:
166 |                 stale += 1
167 |             if epoch % 50 == 0 or stale == 0:
168 |                 log("validation", f"{code} seed {seed} epoch {epoch}: val_auc={validation_auc:.4f} "
169 |                                   f"(best {best['auc']:.4f} at epoch {best['epoch']})")
170 |             if stale >= settings["patience"]:
171 |                 log("train", f"{code} seed {seed}: early stopping at epoch {epoch}")
172 |                 break
173 | 
```

**Dòng 174–185.** Khôi phục trọng số tốt nhất; chấm val và test một lần.

```python
174 |     # Score validation and test once, with the best weights.
175 |     if best["state"] is not None:
176 |         model.load_state_dict(best["state"])
177 |     scores = {}
178 |     for split in ("validation", "test"):
179 |         probabilities = predict(model, x, graph, index[split])
180 |         scores[split] = metrics(split_labels[split], probabilities)
181 |         parts = []
182 |         for name in scores[split]:
183 |             parts.append(f"{name}={scores[split][name]:.4f}")
184 |         log(split, f"{code} seed {seed}: " + ", ".join(parts))
185 | 
```

**Dòng 186–210.** Tạo dòng kết quả và ghi vào results.csv.

```python
186 |     # One row of results.csv.
187 |     row = {}
188 |     row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
189 |     row["scenario"] = code
190 |     row["seed"] = seed
191 |     row["families"] = "+".join(scenario["families"])
192 |     row["features"] = scenario["features"]
193 |     row["hgnn_layers"] = scenario["hgnn_layers"]
194 |     row["use_mlp"] = scenario["use_mlp"]
195 |     row["learn_w"] = scenario["learn_w"]
196 |     for name in ("epochs", "eval_every", "patience", "hidden_dim", "dropout", "learning_rate", "weight_decay"):
197 |         row[name] = settings[name]
198 |     row["best_epoch"] = best["epoch"]
199 |     row["epochs_run"] = epoch
200 |     for name in ("auc", "auprc", "f1"):
201 |         row[f"val_{name}"] = scores["validation"][name]
202 |     for name in scores["test"]:
203 |         row[f"test_{name}"] = scores["test"][name]
204 |     weights = model.weight_summary(scenario["families"])
205 |     for name in weights:
206 |         row[name] = weights[name]
207 |     row["minutes"] = (time.perf_counter() - started_at) / 60
208 |     append_result(row)
209 | 
210 | 
```

**Dòng 211–225.** "all" → `SCENARIOS_ALL`; kiểm tra mã kịch bản.

```python
211 | # Scenario codes to run: --scenario all -> config.SCENARIOS_ALL; otherwise the given codes, checked.
212 | def scenario_codes(values):
213 |     if values == ["all"]:
214 |         codes = list(config.SCENARIOS_ALL)
215 |     else:
216 |         codes = values
217 |     unknown = []
218 |     for code in codes:
219 |         if code not in config.SCENARIOS:
220 |             unknown.append(code)
221 |     if unknown:
222 |         raise SystemExit(f"unknown scenarios {unknown}; choose from {list(config.SCENARIOS)} or 'all'")
223 |     return codes
224 | 
225 | 
```

**Dòng 226–247.** Dòng lệnh và cài đặt.

```python
226 | if __name__ == "__main__":
227 |     parser = argparse.ArgumentParser(description="Run the experiment scenarios: train with early stopping "
228 |                                                  "on validation AUC, then score validation and test at 0.5.")
229 |     all_codes = " ".join(config.SCENARIOS_ALL)
230 |     parser.add_argument("--scenario", nargs="+", default=["M"],
231 |                         help=f"'all' ({all_codes}) or codes from {list(config.SCENARIOS)}")
232 |     parser.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
233 |     parser.add_argument("--epochs", type=int, default=config.TRAIN["epochs"])
234 |     parser.add_argument("--eval-every", type=int, default=config.TRAIN["eval_every"])
235 |     parser.add_argument("--patience", type=int, default=config.TRAIN["patience"])
236 |     parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
237 |     parser.add_argument("--output-dir", default=config.PROCESSED)
238 |     arguments = parser.parse_args()
239 | 
240 |     codes = scenario_codes(arguments.scenario)
241 |     settings = dict(config.TRAIN)
242 |     settings["epochs"] = arguments.epochs
243 |     settings["eval_every"] = arguments.eval_every
244 |     settings["patience"] = arguments.patience
245 |     device = resolve_device(arguments.device)
246 |     log("setup", f"device={device}, scenarios={codes}, seeds={arguments.seeds}, settings={settings}")
247 | 
```

**Dòng 248–255.** Nạp dữ liệu một lần, in số node mỗi split.

```python
248 |     # Read the data once (all families, all columns); every scenario takes its part of it.
249 |     loaded = graph_data.load_graph(arguments.output_dir)
250 |     split_counts = []
251 |     for split_id in range(len(config.SPLITS)):
252 |         count = int(np.sum(loaded["split"] == split_id))
253 |         split_counts.append(f"{config.SPLITS[split_id]}={count:,}")
254 |     log("setup", f"H over {loaded['graph']['num_nodes']:,} nodes: " + ", ".join(split_counts))
255 | 
```

**Dòng 256–265.** Mỗi kịch bản: áp kịch bản, chạy đủ các seed.

```python
256 |     # Scenario by scenario; every seed of a scenario before the next one.
257 |     for code in codes:
258 |         scenario = config.SCENARIOS[code]   # the Vietnamese description is not logged (Windows console)
259 |         families_text = "+".join(scenario["families"])
260 |         log("scenario", f"{code}: families={families_text}, features={scenario['features']}, "
261 |                         f"hgnn_layers={scenario['hgnn_layers']}, use_mlp={scenario['use_mlp']}, "
262 |                         f"learn_w={scenario['learn_w']}")
263 |         data = graph_data.apply_scenario(loaded, config.SCENARIOS[code])
264 |         for seed in arguments.seeds:
265 |             train_one_seed(code, seed, settings, data, device)
```

---

## 11. `10_summary.py` — Tổng hợp

| Dòng | Làm gì |
|---|---|
| 18–19 | Đọc results.csv; trùng (kịch bản, seed) thì **giữ dòng mới nhất** |
| 21–28 | Sắp theo thứ tự kịch bản trong config, rồi theo seed |
| 30–34 | Tỉ lệ trọng số family: w_f / Σ w_f |
| 36–82 | Mỗi kịch bản 1 dòng: ✓/✗ cho từng family, MLP, learn_w; mean ± std (**ddof = 1**) của 9 chỉ số; epoch tốt nhất trung bình; tỉ lệ w; thời gian |
| 85–90 | Cột Δ test AUC so với M |
| 92–95 | Ghi `summary.csv` (UTF-8 có BOM để Excel hiện đúng "±", "✓") và `ket_qua.xlsx` (2 sheet: Tong_hop, Tung_seed; cần `openpyxl`) |

### Mã nguồn có chú giải

File `src/10_summary.py`: 102 dòng, md5 `55710e42484d6ef6cdc162672b98a3e6`. Mỗi dòng code xuất hiện đúng một lần, theo thứ tự.

**Dòng 1–15.** Tiêu đề, import, danh sách chỉ số và trọng số cần tổng hợp.

```python
  1 | # 10. Summarize outputs/summary.csv, outputs/ket_qua.xlsx
  2 | 
  3 | import sys
  4 | from importlib import import_module
  5 | 
  6 | import pandas as pd
  7 | 
  8 | config = import_module("0_config")
  9 | 
 10 | METRICS = ("val_auc", "val_auprc", "val_f1",
 11 |            "test_auc", "test_auprc", "test_accuracy", "test_precision", "test_recall", "test_f1")
 12 | WEIGHTS = ("w_course", "w_object", "w_user", "w_self_loop")
 13 | COMPONENTS = ("course", "object", "user", "self_loop")
 14 | 
 15 | 
```

**Dòng 16–29.** Đọc results.csv; trùng (kịch bản, seed) thì giữ dòng mới nhất; sắp theo thứ tự kịch bản trong config.

```python
 16 | # Read results.csv and write summary.csv and ket_qua.xlsx (one row per scenario, mean ± std).
 17 | def summarize():
 18 |     runs = pd.read_csv(config.RESULTS_CSV)
 19 |     runs = runs.drop_duplicates(["scenario", "seed"], keep="last")
 20 | 
 21 |     order = {}
 22 |     index = 0
 23 |     for code in config.SCENARIOS:
 24 |         order[code] = index
 25 |         index += 1
 26 |     runs["order"] = runs["scenario"].map(order)
 27 |     runs = runs.sort_values(["order", "seed"])
 28 |     runs = runs.drop(columns="order")
 29 |     
```

**Dòng 30–35.** Tỉ lệ trọng số family: w_f / Σ w_f.

```python
 30 |     weights = pd.DataFrame(index=runs.index)
 31 |     for name in WEIGHTS:
 32 |         weights[name] = pd.to_numeric(runs[name], errors="coerce")                # "" -> NaN
 33 |     for name in WEIGHTS:
 34 |         runs[f"{name}_share"] = weights[name] / weights.sum(axis=1)                # w_f / Σ_f w_f
 35 | 
```

**Dòng 36–65.** Mỗi kịch bản 1 dòng: mô tả, ✓/✗ cho family, MLP, learn_w, danh sách seed.

```python
 36 |     rows = []
 37 |     for code, group in runs.groupby("scenario", sort=False):
 38 |         if code in config.SCENARIOS:
 39 |             description = config.SCENARIOS[code]["description"]
 40 |         else:
 41 |             description = ""                                                       # scenario no longer in config
 42 |         families = str(group["families"].iloc[-1]).split("+")
 43 |         row = {}
 44 |         row["scenario"] = code
 45 |         row["description"] = description
 46 |         for name in COMPONENTS:                                                    # ✓ / ✗ columns
 47 |             if name in families:
 48 |                 row[name] = "✓"
 49 |             else:
 50 |                 row[name] = "✗"
 51 |         row["features"] = group["features"].iloc[-1]
 52 |         row["hgnn_layers"] = group["hgnn_layers"].iloc[-1]
 53 |         if bool(group["use_mlp"].iloc[-1]):
 54 |             row["mlp"] = "✓"
 55 |         else:
 56 |             row["mlp"] = "✗"
 57 |         if bool(group["learn_w"].iloc[-1]):
 58 |             row["learn_w"] = "✓"
 59 |         else:
 60 |             row["learn_w"] = "✗"
 61 |         seed_texts = []
 62 |         for seed in group["seed"]:
 63 |             seed_texts.append(str(seed))
 64 |         row["seeds"] = " ".join(seed_texts)
 65 |         row["n"] = len(group)
```

**Dòng 66–83.** mean ± std (ddof = 1) của 9 chỉ số; epoch tốt nhất trung bình; tỉ lệ w; thời gian.

```python
 66 |         for name in METRICS:
 67 |             # mean = (1/n)·Σ x,  std = √(Σ(x − mean)²/(n − 1))  over the n seeds
 68 |             mean = group[name].mean()
 69 |             std = group[name].std(ddof=1)
 70 |             if pd.isna(std):
 71 |                 std = 0.0                                                          # one seed: no std
 72 |             row[name] = f"{mean:.4f} ± {std:.4f}"
 73 |         row["test_auc_mean"] = group["test_auc"].mean()
 74 |         row["best_epoch"] = round(group["best_epoch"].mean())
 75 |         for name in WEIGHTS:
 76 |             share = group[f"{name}_share"].mean()
 77 |             if pd.isna(share):
 78 |                 row[f"{name}_share"] = ""
 79 |             else:
 80 |                 row[f"{name}_share"] = round(share, 4)
 81 |         row["minutes"] = round(group["minutes"].mean(), 1)
 82 |         rows.append(row)
 83 |     summary = pd.DataFrame(rows)
```

**Dòng 84–91.** Cột Δ test AUC so với M.

```python
 84 | 
 85 |     # Δ test AUC = mean test AUC of the scenario − mean test AUC of M.
 86 |     if "M" in summary["scenario"].tolist():
 87 |         reference = summary.loc[summary["scenario"] == "M", "test_auc_mean"].iloc[0]
 88 |         summary.insert(summary.columns.get_loc("test_auc") + 1, "delta_test_auc_vs_M",
 89 |                        (summary["test_auc_mean"] - reference).round(4))
 90 |     summary = summary.drop(columns="test_auc_mean")
 91 | 
```

**Dòng 92–99.** Ghi `summary.csv` (UTF-8 BOM) và `ket_qua.xlsx` (2 sheet), in bảng.

```python
 92 |     summary.to_csv(config.OUTPUTS / "summary.csv", index=False, encoding="utf-8-sig")   # BOM: Excel shows "±", "✓"
 93 |     with pd.ExcelWriter(config.OUTPUTS / "ket_qua.xlsx") as writer:
 94 |         summary.to_excel(writer, sheet_name="Tong_hop", index=False)
 95 |         runs.to_excel(writer, sheet_name="Tung_seed", index=False)
 96 |     print(summary.drop(columns="description").to_string(index=False))
 97 |     print(f"Saved {config.OUTPUTS / 'summary.csv'} and {config.OUTPUTS / 'ket_qua.xlsx'}")
 98 | 
 99 | 
```

**Dòng 100–102.** Chạy chính; in được "±" trên console Windows.

```python
100 | if __name__ == "__main__":
101 |     sys.stdout.reconfigure(encoding="utf-8")   # print "±" and "✓" also on a Windows console
102 |     summarize()
```

---

## 12. Bảng "muốn sửa X thì sửa ở đâu"

| Muốn đổi | Sửa ở | Chạy lại từ bước |
|---|---|---|
| Cửa sổ ngày (35) | `0_config.py:30` | 2 (xóa 3 CSV trước) |
| Danh sách action | `0_config.py:31-47` | 2 (xóa CSV) |
| Tỉ lệ / seed của split | `0_config.py:28-29` | 2 (xóa CSV) |
| Cách chuẩn hóa hành vi (log1p, z-score) | `3_features.py:224-227, 240-241` | 3 |
| Thêm cột feature | config 75–95 + `3_features.py` (`feature_metadata`, `build_split_features`, `feature_columns`) + `FEATURE_COUNT` trong `baseline/ML/*.py` | 3 |
| Thêm loại hyperedge | config 106 + `4_hypergraph.py` (hàm mới, dòng 110) + kịch bản trong config | 4 |
| Bỏ hyperedge kích thước 1 hay không | `4_hypergraph.py:112-115` | 4 |
| Công thức lan truyền | `6_hgnn.py:59-66` | 9 |
| Siêu tham số | `0_config.py:137-146` hoặc tham số dòng lệnh | 9 |
| Ngưỡng 0.5 | `0_config.py:136` (và các file baseline có hằng `THRESHOLD` riêng) | 9 |
| Chỉ số / cột kết quả | `9_train.py:21-30, 57-71, 187-207` + `10_summary.py:10-11` | 9 |

---

## 13. Những điểm dễ bỏ sót (thầy rà khi đọc từng dòng)

| # | Điểm | Ở đâu | Ảnh hưởng |
|---|---|---|---|
| 1 | `2_preprocess` **bỏ qua** khi đã có 3 CSV | `2_preprocess.py:182-188` | Đổi luật lọc mà không xóa CSV thì code vẫn dùng dữ liệu cũ, không báo gì |
| 2 | Patience tính theo **số lần chấm**, không theo epoch | `9_train.py:158-172` | 40 lần × 5 = 200 epoch. Khi viết bài phải ghi "patience 200 epochs" (hoặc "40 validation checks every 5 epochs") |
| 3 | `distinct_objects` (cột 36) đếm object của **mọi** action, còn hyperedge Object chỉ từ video/assignment/forum | `3_features.py:135-137` và `4_hypergraph.py:38` | Không sai, nhưng hai định nghĩa "object" khác nhau; nên ghi rõ trong bài |
| 4 | Hyperedge User gồm **mọi** enrollment của người học, kể cả khóa mở **sau** khóa đang dự đoán | `4_hypergraph.py:80-90` | Đây là quy tắc "any" em đã chọn. Bộ nhớ ngày 03/10 ghi: bản "không thấy tương lai" không tăng AUC test. Nên nêu trong phần giới hạn |
| 5 | Transductive: feature của val/test tham gia lan truyền | `4_hypergraph.py:99-108`, `9_train.py:145` | Đúng chuẩn HGNN+/AllSet; nhãn val/test không vào loss |
| 6 | Giá trị phân loại lạ (gender, education, category mới) → **báo lỗi** chứ không âm thầm bỏ qua | `3_features.py:93` | An toàn; khi đổi dataset phải thêm vào config |
| 7 | Không tạo node cho enrollment không có nhãn | `2_preprocess.py:88-89` | Đúng ý |
| 8 | CPU nhiều luồng: phép nhân sparse cho sai khác khoảng 1e-7 giữa các lần chạy cùng seed | `6_hgnn.py:61-66` | Muốn giống hệt từng bit thì `torch.set_num_threads(1)` (đã kiểm tra hôm 07/10) |
| 9 | `download_oulad` là hàm rỗng | `1_download.py:26-27` | Không ảnh hưởng; có thể xóa |
| 10 | Lỗi chính tả trong chú thích: "RRCALL" | `9_train.py:60` | Chỉ là chú thích |
| 11 | `labels` lưu float32 | `4_hypergraph.py:137` | Loss dùng float là đúng; baseline phải ép về int (LR/SVM đã làm) |
| 12 | Mỗi lần chạy `9_train` **nối thêm** vào results.csv | `9_train.py:86-102` | Chạy lại cùng (kịch bản, seed) thì `10_summary` lấy dòng mới nhất; file đổi cột thì phải đổi tên file cũ |
