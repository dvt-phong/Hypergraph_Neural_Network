# Dữ liệu vào/ra theo từng file và từng function

Tài liệu này đi qua pipeline `src/1_download.py` → `src/8_train.py`. Với mỗi file
có hai phần:

1. **Kết thúc file thu được dữ liệu gì**: file nào, shape, ý nghĩa, kèm mẫu.
2. **Từng function**: nhận vào gì, trả ra gì, kèm mẫu.

Mẫu được lấy từ dữ liệu thật trong `data/processed/simple` (chạy ngày
26/09/2026, cấu hình `k = 10`, `k_max = 20`). Mẫu nào là ví dụ tự dựng để dễ
nhìn thì ghi rõ **(toy)**; mẫu minh hoạ vì chưa có lần chạy thật thì ghi
**(minh hoạ)**.

```text
1_download   → data/raw/xuetangx/{prediction_data.tar.gz, user_info.csv, course_info.csv}
2_preprocess → data/processed/simple/{train,validation,test}.csv      (mỗi event một dòng)
3_features   → data/processed/simple/{split}/X.npy + feature_names.csv (mỗi node một hàng, 92 cột)
4_hypergraph → data/processed/simple/hypergraph.npz                    (H0 của train + kNN)
5,6,7        → không ghi file; là model và loss được 8_train gọi
8_train      → outputs/runs/<run>.pt, outputs/reports/<run>_{train,test}.json
```

Tổng quan số lượng sau khi chạy:

| Split | Enrollment (node) | Dòng event trong CSV | Tỉ lệ dropout |
|---|---|---|---|
| train | 126 354 | 22 430 491 | 75,8% |
| validation | 31 589 | 5 661 741 | 75,8% (cùng nguồn `train_truth`) |
| test | 67 699 | 12 469 713 | 75,8% |

---

## 0. `0_config.py`: hằng số dùng chung

Không đọc hay ghi dữ liệu. Kết thúc file có các hằng số, các file khác import để
dùng:

| Hằng số | Giá trị | Dùng ở |
|---|---|---|
| `RAW`, `PROCESSED`, `RUNS`, `REPORTS` | `data/raw/xuetangx`, `data/processed/simple`, `outputs/runs`, `outputs/reports` | mọi file |
| `SEEDS` | `(1, 11, 111, 1111, 11111)` | `8_train`, `scripts/` |
| `SPLITS` | `("train", "validation", "test")` | 2, 3 |
| `SPLIT_SEED`, `TRAIN_RATIO` | `1`, `0.80` | 2 |
| `OBSERVATION_DAYS` | `35` | 2, 3 |
| `ACTIONS` | 23 action, chia 4 nhóm video/assignment/forum/web_page | 2, 3 |
| `OBJECT_ACTIONS` | action → `"video"` / `"assignment"` / `"forum"` (bỏ web_page) | 4 |
| `BEHAVIOR_FEATURE_SLICE` | `slice(0, 58)` | 3, 4 |
| `USER_FEATURE_START`, `COURSE_FEATURE_START`, `TOTAL_FEATURE_COUNT` | `58`, `73`, `92` | 3 |
| `EDGE_FAMILIES` | `("course", "object", "behavioral", "self_loop")` → 0, 1, 2, 3 | 4, 6, 7 |
| `MEMBERSHIP_CHUNK_SIZE` | `1_000_000` | 5, 6 |

---

## 1. `1_download.py`

### Kết thúc file thu được

```text
data/raw/xuetangx/
├── prediction_data.tar.gz   313 MB, bên trong có 4 file CSV:
│     prediction_log/train_log.csv    4,2 GB   log hành vi của các enrollment train
│     prediction_log/train_truth.csv  157 943 enrollment + nhãn
│     prediction_log/test_log.csv     1,9 GB
│     prediction_log/test_truth.csv   67 699 enrollment + nhãn
├── user_info.csv            117 MB
└── course_info.csv          6 410 khoá học
```

Mẫu (dữ liệu thật):

```text
train_log.csv
enroll_id,username,course_id,session_id,action,object,time
772,5981,course-v1:TsinghuaX+70800232X+2015_T2,d8a9b787...,click_about,,2015-09-27T15:42:59
773,1544995,course-v1:TsinghuaX+70800232X+2015_T2,2f02b86e...,pause_video,3dac5590435e43b3a65a9ae7426c16db,2015-10-19T19:37:42

train_truth.csv          user_info.csv                    course_info.csv
enroll_id,truth          user_id,gender,education,birth   id,course_id,start,end,course_type,category
772,1                    631,male,High,1997.0             6561,course-v1:CPVS+CPVS-HDLSC001+20160901,2016-11-16 08:00:00,2016-12-31 23:30:00,0,
773,1                    2631,male,Bachelor's,1990.0
```

`truth = 1` nghĩa là dropout.

### `download(raw_dir=config.RAW)`

| | |
|---|---|
| **Nhận** | `raw_dir`: thư mục đích, mặc định `data/raw/xuetangx` |
| **Làm** | Với mỗi file trong `config.DOWNLOAD_FILES`: nếu đã có thì bỏ qua, chưa có thì tải về |
| **Trả** | `None`; sinh ra 3 file raw ở trên |

---

## 2. `2_preprocess.py`

### Kết thúc file thu được

Ba file CSV, **mỗi dòng là một event** của một enrollment trong 35 ngày đầu khoá
học. Thông tin user và course được lặp lại trên mọi dòng, để các bước sau chỉ
cần đọc một file.

```text
data/processed/simple/train.csv       126 354 node, 22 430 491 dòng event
data/processed/simple/validation.csv   31 589 node,  5 661 741 dòng
data/processed/simple/test.csv         67 699 node, 12 469 713 dòng
```

| Cột | Ý nghĩa |
|---|---|
| `node_id` | chỉ số node **trong split**, liên tục `0 … N−1` |
| `enroll_id`, `user_id`, `course_id` | ID gốc |
| `label` | 1 = dropout |
| `gender`, `education`, `birth` | từ `user_info.csv` (có thể rỗng) |
| `course_start`, `course_end`, `category` | từ `course_info.csv` |
| `action`, `object_id` | event |
| `course_day` | số ngày tính từ `course_start`, trong khoảng `0 … 34` |

Mẫu (`train.csv`, thật):

```text
node_id,enroll_id,user_id,course_id,label,gender,education,birth,course_start,course_end,category,action,object_id,course_day
0,772,5981,course-v1:TsinghuaX+70800232X+2015_T2,1,male,Master's,1989.0,2015-09-25 08:00:00,2016-01-06 08:00:00,art,click_about,,2
0,772,5981,course-v1:TsinghuaX+70800232X+2015_T2,1,male,Master's,1989.0,2015-09-25 08:00:00,2016-01-06 08:00:00,art,click_info,,2
1,774,1072798,course-v1:TsinghuaX+70800232X+2015_T2,1,,,,2015-09-25 08:00:00,2016-01-06 08:00:00,art,seek_video,3169d758ee2d4262b07f0113df743c42,25
```

Enrollment **không có event nào** trong 35 ngày vẫn được giữ, dưới dạng một dòng
có `action`, `object_id`, `course_day` rỗng:

```text
124475,840,558112,course-v1:TsinghuaX+70800232X+2015_T2,1,,,,2015-09-25 08:00:00,2016-01-06 08:00:00,art,,,
```

### `read_csv(path)`

| | |
|---|---|
| **Nhận** | đường dẫn một file CSV UTF-8 |
| **Trả** | generator, mỗi phần tử là `dict` một dòng (mọi giá trị đều là `str`) |
| **Mẫu** | `{"user_id": "631", "gender": "male", "education": "High", "birth": "1997.0"}` |

### `write_csv(path, columns, rows)`

| | |
|---|---|
| **Nhận** | `path`, tên cột (`tuple`), `rows` (iterable các `tuple`) |
| **Trả** | `None`; ghi file CSV |
| **Mẫu** | `write_csv(p, ("feature_index", "feature_name", "source"), [(0, "activity_day_0", "course_day=0")])` |

### `load_nodes(path)`

| | |
|---|---|
| **Nhận** | một split CSV (mỗi event một dòng) |
| **Trả** | `list[dict]`, **mỗi node một phần tử**, sắp theo `node_id`; lấy dòng đầu tiên của mỗi node |
| **Mẫu** | `load_nodes("validation.csv")[0]` → `{"node_id": "0", "enroll_id": "773", "user_id": "1544995", "course_id": "course-v1:TsinghuaX+70800232X+2015_T2", "label": "1", "gender": "", "education": "", "birth": "", "course_start": "2015-09-25 08:00:00", ...}` |

Phần tử thứ `i` ứng với hàng `i` của `X.npy`.

### `read_prediction_data(prediction_data_path, selected_names)`

| | |
|---|---|
| **Nhận** | đường dẫn `prediction_data.tar.gz`; tập tên file cần đọc, ví dụ `("train_truth.csv", "test_truth.csv")` |
| **Trả** | generator `(tên_file, dict_dòng)`, đọc tuần tự, không giải nén ra đĩa |
| **Mẫu** | `("train_truth.csv", {"enroll_id": "772", "truth": "1"})` |

### `split_train_enrollments(enrollment_ids)`

| | |
|---|---|
| **Nhận** | danh sách `enroll_id` của `train_truth` (đã sắp) |
| **Trả** | `dict enroll_id → "train" / "validation"`; xáo bằng `SPLIT_SEED = 1`, lấy 80% đầu làm train |
| **Mẫu (toy)** | `split_train_enrollments(range(10))` → `{6: "train", 8: "train", 9: "train", 7: "train", 5: "train", 3: "train", 0: "train", 4: "train", 1: "validation", 2: "validation"}` |

Với dữ liệu thật: 157 943 enrollment → 126 354 train + 31 589 validation.

### `stream_events(prediction_data_path, output_dir, users, courses, labels, split_by_enrollment)`

| | |
|---|---|
| **Nhận** | `users: {user_id(int): dict}`, `courses: {course_id: dict}`, `labels: {enroll_id: 0/1}`, `split_by_enrollment: {enroll_id: split}` |
| **Làm** | Đọc `train_log` và `test_log`. Bỏ event có action ngoài `ACTIONS` hoặc nằm ngoài ngày 0–34. Gán `node_id` theo thứ tự xuất hiện trong mỗi split. Sau đó bổ sung các enrollment không có event nào |
| **Trả** | `None`; ghi 3 file CSV ở trên, in ra số enrollment không có event |
| **Mẫu dòng vào** | `772,5981,course-v1:...,d8a9...,click_about,,2015-09-27T15:42:59` |
| **Mẫu dòng ra** | `0,772,5981,course-v1:...,1,male,Master's,1989.0,2015-09-25 08:00:00,...,art,click_about,,2` (`course_day = 27/09 − 25/09 = 2`) |

### `preprocess(raw_dir, output_dir)`

| | |
|---|---|
| **Nhận** | thư mục raw, thư mục output |
| **Làm** | nạp `users`, `courses`, nhãn → chia train/validation (test giữ nguyên split chính thức) → `stream_events` |
| **Trả** | `None` |

**Lưu ý:** nếu cả ba file `{split}.csv` đã tồn tại, hàm **bỏ qua** toàn bộ bước
này ([2_preprocess.py:209-211](../src/2_preprocess.py#L209-L211)). Muốn chạy lại
bước 2 thì phải xoá ba file CSV cũ.

---

## 3. `3_features.py`

### Kết thúc file thu được

```text
data/processed/simple/train/X.npy        (126 354, 92) float32
data/processed/simple/validation/X.npy   ( 31 589, 92) float32
data/processed/simple/test/X.npy         ( 67 699, 92) float32
data/processed/simple/feature_names.csv  92 dòng: feature_index, feature_name, source
```

Bố cục 92 cột: behavior 0–57 (35 cột ngày + 23 cột action, đã `log1p` và
z-score) | user 58–72 (gender 4, education 9, age 2) | course 73–91 (category 19).

Mẫu: hàng 0 của `train/X.npy` (enroll 772, nam, Thạc sĩ, sinh năm 1989, khoá
category art):

```text
cột 0–34   [-0.2509 -0.3118  0.879 -0.2756 ...]   ngày 2 có hoạt động → 0.879, các ngày khác âm (dưới mức trung bình)
cột 35–57  [-0.6606 -0.9132 -1.0625 ...]
cột 58–91  gender_male=1, education_masters=1, age_at_course_start=0.278, category_art=1, còn lại 0
```

Kiểm tra trên train: mean của behavior xấp xỉ 0 (lệch lớn nhất 2,7e-5), std xấp
xỉ 1. Riêng cột `action_web_page_7 (close_info)` có std = 0, vì action này không
xuất hiện trong train. Có **71,6%** node thiếu tuổi (`age_missing = 1`).

### `feature_columns(feature_set)`

| | |
|---|---|
| **Nhận** | `"behavior"`, `"behavior_user"`, `"behavior_course"` hoặc `"full"` |
| **Trả** | chỉ số cột để cắt `X` |
| **Mẫu** | `"behavior"` → `slice(0, 58)` (58 cột); `"behavior_user"` → `slice(0, 73)`; `"behavior_course"` → mảng 77 chỉ số `[0 … 57, 73 … 91]`; `"full"` → `slice(0, 92)` |

### `feature_metadata()`

| | |
|---|---|
| **Nhận** | không có |
| **Trả** | `list` 92 phần tử `(feature_name, source)`, đúng thứ tự cột |
| **Mẫu** | `[0] ("activity_day_0", "course_day=0")`, `[35] ("action_video_1", "seek_video")`, `[71] ("age_at_course_start", "course start year - birth year")`, `[91] ("category_other", "category outside vocabulary")` |

### `set_one_hot(feature_matrix, row_index, value, vocabulary, feature_start)`

| | |
|---|---|
| **Nhận** | ma trận, hàng, giá trị chuỗi, từ điển giá trị hợp lệ, cột bắt đầu của nhóm |
| **Làm** | đặt 1 vào một cột: giá trị nằm trong từ điển → cột của nó; rỗng → cột `missing`; lạ → cột `other` |
| **Trả** | `None` (sửa trực tiếp trên ma trận) |
| **Mẫu** | trên `user_context` 15 cột: `"Master's"` → 1 ở cột 8 (education thứ 5); gender `""` → 1 ở cột 2 (`gender_missing`); gender `"unknown"` → 1 ở cột 3 (`gender_other`) |

### `numeric_statistics(train_values)`

| | |
|---|---|
| **Nhận** | mảng số của train, có thể chứa `NaN` |
| **Trả** | `(median, mean, std)`; mean và std tính **sau khi** điền `NaN` bằng median |
| **Mẫu (toy)** | `[20, NaN, 25, 40, NaN]` → median 25 → điền thành `[20, 25, 25, 40, 25]` → `(25.0, 27.0, 6.7823)` |

### `scale_numeric(raw_values, statistics)`

| | |
|---|---|
| **Nhận** | mảng số của một split, bộ thống kê của train |
| **Trả** | `(giá_trị_đã_chuẩn_hoá, cờ_thiếu)`, cả hai đều float32 |
| **Mẫu (toy)** | với thống kê ở trên: `[20, NaN, 25, 40, NaN]` → `([-1.0321, -0.2949, -0.2949, 1.9167, -0.2949], [0, 1, 0, 0, 1])` |

### `build_split_features(data_path)`

| | |
|---|---|
| **Nhận** | một split CSV (mỗi event một dòng) |
| **Trả** | `behavior (N, 58) int32` (số đếm thô), `user (N, 15) float32` (hai cột tuổi còn bằng 0), `course (N, 19) float32`, `ages (N,) float64` (tuổi thô, `NaN` nếu thiếu hoặc nằm ngoài 10–100) |
| **Mẫu** | chạy trên 400 dòng đầu của `validation.csv` (13 node): node 0 có `behavior` khác 0 tại `{24: 21, 35: 6, 36: 4, 37: 3, 39: 1, 51: 2, 52: 2, 53: 2, 56: 1}`, tức 21 event vào ngày 24, gồm seek 6, play 4, pause 3 video…; `user[0]` = 1 ở `gender_missing` và `education_missing`; `ages = [nan 21 nan nan nan nan 20 22 nan nan 25 nan 19]` |

### `build_features(output_dir)`

| | |
|---|---|
| **Nhận** | thư mục chứa `{split}.csv` |
| **Làm** | `build_split_features` cho 3 split → fit `log1p` + mean/std behavior và thống kê tuổi **trên train** → transform cả 3 split → ghép `behavior | user | course` |
| **Trả** | `{"train": Path(.../train/X.npy), "validation": ..., "test": ...}` và ghi `feature_names.csv` |

Giải thích từng dòng của hàm này có trong phần trao đổi trước; cột nào nằm ở
đâu thì xem bảng bố cục ở trên.

---

## 4. `4_hypergraph.py`

### Kết thúc file thu được

`data/processed/simple/hypergraph.npz` (19 MB):

| Khoá | Shape, dtype | Mẫu |
|---|---|---|
| `node_ids` | `(3 075 381,)` int64 | `[26068 26069 26070 …]` |
| `edge_ids` | `(3 075 381,)` int64, đã sắp tăng | `[0 0 0 …]` |
| `edge_family` | `(148 348,)` int64 | `[0 0 0 …]` (0 = course) |
| `edge_keys` | `(148 348,)` str | `"CAU/08112500x/2015_T2"`, `"…|assignment|0054247a…"`, `"0"` |
| `train_neighbors` | `(126 354, 20)` int64 | hàng 0: `[27728 12876 13066 4988 248 43011 957 47935 938 12591 │ 34981 3606 …]` |
| `validation_neighbors` | `(31 589, 20)` int64 | hàng 0: `[25927 44607 65749 …]` |
| `test_neighbors` | `(67 699, 20)` int64 | hàng 0: `[36154 112957 31866 …]` |
| `k` | scalar | `10` |

Ở `train_neighbors`, 10 cột đầu (trước dấu `│`) là thành viên Behavioral, 10 cột
sau là ứng viên ΔH của HSL.

Thống kê `H0` (thật, chưa có self-loop):

| Family | Số hyperedge | Kích thước min / median / mean / max | Số membership |
|---|---|---|---|
| course | 247 | 151 / 377 / 511,6 / 2 361 | 126 354 |
| object | 21 747 | 2 / 27 / 71,7 / 1 893 | 1 559 133 |
| behavioral | 126 354 | 11 / 11 / 11 / 11 | 1 389 894 |
| **tổng** | **148 348** | | **3 075 381** |

Mẫu một hyperedge mỗi loại:

```text
course      edge 0      key "CAU/08112500x/2015_T2"                         231 node: [26068 26069 26070 …]
object      edge 247    key "CAU/08112500x/2015_T2|assignment|0054247a…"     11 node: [26092 26160 26165 …]
behavioral  edge 21994  key "0"   (anchor = node 0)                         11 node: [0 27728 12876 13066 4988 248 43011 957 47935 938 12591]
```

### `resolve_device(device_name)`

| | |
|---|---|
| **Nhận** | `"auto"`, `"cpu"` hoặc `"cuda"` |
| **Trả** | `torch.device`; `"auto"` → cuda nếu có |

### `nearest_train_neighbors(train_features, query_features, count, *, exclude_self, device, batch_size)`

| | |
|---|---|
| **Nhận** | `train_features (N_train, 92)`, `query_features (N_q, 92)`, `count = k_max = 20` |
| **Làm** | cosine trên 58 cột behavior, tính theo batch, lấy top-`count` |
| **Trả** | `(N_q, count)` int64, chỉ số **train**, sắp theo cosine giảm dần |
| **Mẫu** | 3 query validation đầu → `[[25927 44607 65749 …], [114825 105290 83559 …], [21979 69385 114884 …]]`, trùng khớp với `validation_neighbors` đã lưu |

### `read_object_events(events_path)`

| | |
|---|---|
| **Nhận** | một split CSV |
| **Trả** | generator `(node_id, object_key)` cho **mỗi event** video/assignment/forum có `object_id` (một node có thể lặp lại nhiều lần) |
| **Mẫu** | `(0, "course-v1:TsinghuaX+70800232X+2015_T2|video|3dac5590435e43b3a65a9ae7426c16db")` |

### `build_train_hyperedges(train_nodes, train_events_path, neighbors, k)`

| | |
|---|---|
| **Nhận** | `train_nodes` (từ `load_nodes`), `train.csv`, `train_neighbors (N, 20)`, `k` |
| **Trả** | `dict` gồm `node_ids`, `edge_ids`, `edge_family`, `edge_keys` (chưa có self-loop) |

**Mẫu (toy):** 3 node, node 0 và 1 thuộc C1, node 2 thuộc C2; node 0 và 1 cùng
xem video v1; `k = 1`, neighbors `[[1,2],[0,2],[0,1]]`:

```text
node_ids    [0 1 │ 0 1 │ 0 1 │ 1 0 │ 2 0]
edge_ids    [0 0 │ 1 1 │ 2 2 │ 3 3 │ 4 4]
edge_family [0, 1, 2, 2, 2]
edge_keys   ["C1", "C1|video|v1", "0", "1", "2"]
```

C2 và `C2|assignment|p7` chỉ có 1 thành viên nên bị lọc bỏ; event `click_info` bị
bỏ vì thuộc web_page.

### `build_hypergraph(output_dir, *, k, k_max, device_name, batch_size)`

| | |
|---|---|
| **Nhận** | `train.csv`, `{split}/X.npy` |
| **Làm** | kNN cho 3 split (kho tìm kiếm luôn là train) → `build_train_hyperedges` → lưu file |
| **Trả** | `None`; ghi `hypergraph.npz` |

### `add_self_loops(graph)`

| | |
|---|---|
| **Nhận** | `graph` có `num_nodes`, `node_ids`, `edge_ids`, `edge_family` |
| **Trả** | `graph` mới, thêm ở **cuối** mỗi node một hyperedge `{v}` với family 3 |
| **Mẫu (toy, nối tiếp ví dụ trên)** | `node_ids [… 0 1 2]`, `edge_ids [… 5 6 7]`, `edge_family [0 1 2 2 2 3 3 3]` |

### `load_train_graph(output_dir, *, feature_set)`

| | |
|---|---|
| **Nhận** | thư mục processed, feature set |
| **Trả** | `{"features": (126354, 92) float32, "labels": (126354,) float32, "graph": dict}` |

`graph` sau khi thêm self-loop (thật):

| Khoá | Shape | Mẫu |
|---|---|---|
| `num_nodes` | int | `126354` |
| `node_ids`, `edge_ids` | `(3 201 735,)` = 3 075 381 + 126 354 self-loop | đuôi: `node_ids [126351 126352 126353]`, `edge_ids [274699 274700 274701]` |
| `edge_family` | `(274 702,)` | đuôi `[3 3 3]` |
| `candidate_edge_ids` | `(126 354,)` | `[21994 21995 …]`: id các Behavioral hyperedge |
| `candidate_node_ids` | `(126 354, 10)` | hàng 0: `[34981 3606 39950 839 45260 13012 56753 12758 8427 586]` = `train_neighbors[0, 10:]` |

Tỉ lệ nhãn: 75,82% dropout.

### `load_evaluation_split(output_dir, *, split_name, feature_set)`

| | |
|---|---|
| **Nhận** | `"validation"` hoặc `"test"` |
| **Trả** | `dict` chứa mọi thứ cần để dựng local graph cho từng target |

Mẫu với `validation` (thật):

| Khoá | Giá trị |
|---|---|
| `split_name` | `"validation"` |
| `nodes` | list 31 589 dict; `nodes[0]["course_id"] = "course-v1:TsinghuaX+70800232X+2015_T2"`, `label = "1"` |
| `features` | `(31589, 92)` |
| `train_features` | `(126354, 92)` |
| `members_by_key` | dict 21 994 key (course + object) → mảng train member |
| `object_keys` | dict 27 953 node → tập object key; `object_keys[0]` = 1 video |
| `neighbors` | `(31589, 20)` |
| `k` | `10` |

### `build_local_graph(split_data, target_id)`

| | |
|---|---|
| **Nhận** | kết quả `load_evaluation_split`, chỉ số target |
| **Trả** | `{"features": (n, 92), "graph": dict, "label": int}`; node local 0 là target, các node còn lại là train |

Mẫu với target 0 của validation (thật):

```text
features (754, 92)   label 1
graph.num_nodes      754          = target + 753 train node
graph.edge_family    [0 1 2 3 3 3 …]   3 hyperedge thật + 754 self-loop = 757
kích thước           course 734, object 541, behavioral 11, self-loop 1 × 754
graph.node_ids       (2040,)      = 734 + 541 + 11 + 754
candidate_edge_ids   [2]          (Behavioral hyperedge của target)
candidate_node_ids   [[741 730 740 729 739 734 736 732 731 745]]   (chỉ số local)
```

### `merge_local_graphs(local_graphs)`

| | |
|---|---|
| **Nhận** | list các local graph |
| **Trả** | `(features, merged_graph, target_rows, labels)`; các graph đặt cạnh nhau, không có hyperedge nối giữa chúng |
| **Mẫu** | target 0 và 1 → `features (1508, 92)`, `target_rows [0 754]`, `labels [1 1]`, `node_ids (4712,)`, `edge_family (1516,)`, `candidate_node_ids (2, 10)` |

---

## 5. `5_model.py`

### Kết thúc file thu được

Không ghi file. File này cung cấp class `HSLModel`; `8_train` tạo model và gọi
forward. Cấu trúc với cấu hình mặc định (`hidden_dim = 128`):

```text
HSLModel
├── layer1       Linear(92 → 128)
├── layer2       Linear(128 → 128)
├── classifier   Linear(128 → 1)
└── structure_learner (6_hsl.StructureLearner)
    ├── edge_scorer        Linear(128 + 4 → 32) → ReLU → Linear(32 → 1)
    ├── membership_node    Linear(128 → 32)
    ├── membership_edge    Linear(128 → 32, không bias)
    └── membership_output  Linear(32 → 1)
```

### `graph_to_device(graph, device)`

| | |
|---|---|
| **Nhận** | graph NumPy (từ `load_train_graph` hoặc `merge_local_graphs`) |
| **Trả** | cùng các khoá nhưng là tensor int64 trên `device`, thêm `num_edges` |
| **Mẫu** | `{"num_nodes": 1508, "node_ids": (4712,) int64, "edge_ids": (4712,), "edge_family": (1516,), "candidate_edge_ids": (2,), "candidate_node_ids": (2, 10), "num_edges": 1516}` |

### `weighted_sum_chunk(...)` / `weighted_sum(source, weights, source_ids, target_ids, target_count)`

| | |
|---|---|
| **Nhận** | `source (S, D)`, `weights (M,)`, `source_ids (M,)`, `target_ids (M,)`, `target_count` |
| **Trả** | `(target_count, D)` với `output[t] = Σ weights[m] · source[source_ids[m]]` trên các `m` có `target_ids[m] = t` |
| **Mẫu (toy)** | `source = [[1],[2],[3]]`, `weights = [1,1,1]`, `source_ids = [0,1,2]`, `target_ids = [0,0,1]`, `target_count = 2` → `[[3],[3]]` |

Hàm được dùng theo hai chiều: node → hyperedge (`Hᵀ`) và hyperedge → node (`H`).

### `hgnn_propagate(x, graph, weights)`

| | |
|---|---|
| **Nhận** | `x (N, D)`, graph, `weights (M,)` (bằng 1 với `H0`, bằng 0/1 với `H*`) |
| **Trả** | `(N, D)` = `Dv^-1/2 H De^-1 Hᵀ Dv^-1/2 x` |
| **Mẫu (toy)** | 3 node chung một course hyperedge, mỗi node có self-loop, `x = [1, 2, 3]` → `[1.5, 2.0, 2.5]`, tức mỗi node nhận `2/3` feature của chính nó + `1/6` của mỗi node còn lại |
| **Mẫu (thật)** | `hgnn_propagate(layer1(x), merged, ones)` → `(1508, 128)` |

### `hyperedge_means(z, graph)`

| | |
|---|---|
| **Nhận** | `z (N, D)`, graph `H0` |
| **Trả** | `(E, D)`: trung bình embedding của các thành viên mỗi hyperedge |
| **Mẫu** | `z0 (1508, 128)` → `(1516, 128)` |

### `HSLModel.encode(x, graph, weights)`

| | |
|---|---|
| **Nhận** | `x (N, 92)`, graph, `weights` |
| **Trả** | `(N, 128)`: Linear → propagate → ReLU → Dropout → Linear → propagate → ReLU |

### `HSLModel.forward(x, graph)`

| | |
|---|---|
| **Nhận** | `x (N, 92)`, graph đã qua `graph_to_device` |
| **Trả** | `{"logits": (N,), "z0": (N, 128), "z_star": (N, 128), "structure": dict}` |

Mẫu trên 2 local graph (model chưa train, `seed = 1`):

```text
train mode: logits của 2 target [-0.0817, -0.1047]
            structure {"kept_course": 0.963, "kept_object": 0.949, "kept_behavioral": 1.0, "added": 4}
eval mode:  xác suất 2 target [0.4786, 0.4743]
            structure {"kept_course": 1.0, "kept_object": 1.0, "kept_behavioral": 1.0, "added": 4}
```

Nếu tắt HSL (`--no-hsl`): `z_star = z0` và `structure = {}`.

---

## 6. `6_hsl.py`

### Kết thúc file thu được

Không ghi file. File này cung cấp `StructureLearner`, được `HSLModel.forward` gọi
để dựng `H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I`.

### `keep_mask(logits, temperature, training)`

| | |
|---|---|
| **Nhận** | logit giữ lại, nhiệt độ Gumbel (0,4), cờ train |
| **Trả** | mask 0/1 cùng shape. Khi train: rút ngẫu nhiên (Gumbel straight-through). Khi eval: `logit > 0` |
| **Mẫu** | `[-2, 0, 3]` → train (một lần rút) `[0, 1, 1]`; eval `[0, 0, 1]` (logit 0 bị bỏ vì điều kiện là `>` chặt) |

### `StructureLearner.implicit_connections(z0, edge_representations, graph)`

| | |
|---|---|
| **Nhận** | `z0 (N, 128)`, `h_e (E, 128)`, graph có `candidate_*` |
| **Trả** | `(added_nodes, added_edges)`: với mỗi Behavioral hyperedge, `add_per_edge = 2` ứng viên có cosine cao nhất |
| **Mẫu** | 2 local graph → `added_nodes [729 732 1495 1498]`, `added_edges [2 2 761 761]` (2 node cho mỗi Behavioral hyperedge) |

### `StructureLearner.membership_logits(z0, edge_representations, node_ids, edge_ids)`

| | |
|---|---|
| **Nhận** | `z0`, `h_e`, danh sách membership |
| **Trả** | `(M,)`: logit giữ từng membership, `MLP([z_v ‖ h_e])` |
| **Mẫu** | `(4712,)`, lúc mới khởi tạo `[2.964, 2.963, 2.967, …]` ≈ 3, tức xác suất giữ ≈ 0,95 |

### `StructureLearner.forward(z0, edge_representations, graph)`

| | |
|---|---|
| **Nhận** | `z0 (N, 128)`, `h_e (E, 128)`, graph `H0` |
| **Trả** | `(refined_graph, weights, summary)` |
| **Mẫu** | `node_ids` 4 712 → 4 716 (thêm 4 từ ΔH); `weights (4716,)` gồm các giá trị 0/1; summary `{"kept_course": 0.956, "kept_object": 0.957, "kept_behavioral": 0.955, "added": 4}` |

`refined_graph` giữ nguyên `edge_family`; chỉ `node_ids` và `edge_ids` dài
thêm. Self-loop luôn có weight bằng 1.

### `StructureLearner.summary(graph, weights, h0_count)`

| | |
|---|---|
| **Nhận** | graph `H0`, weights của `H*`, số membership của `H0` |
| **Trả** | tỉ lệ membership `H0` được giữ theo family (không tính self-loop) + số membership ΔH được giữ |
| **Mẫu** | `{"kept_course": 0.956, "kept_object": 0.957, "kept_behavioral": 0.955, "added": 4}` |

---

## 7. `7_losses.py`

### Kết thúc file thu được

Không ghi file. File này cung cấp hàm loss, `8_train` gọi mỗi epoch:
`L = BCE(pos_weight) + λ · L_CL`.

### `positive_class_weight(labels)`

| | |
|---|---|
| **Nhận** | `labels (N,)` 0/1 của train |
| **Trả** | `#âm / #dương` |
| **Mẫu** | train thật → `0.3189` (vì 75,8% là dropout = lớp dương, nên lớp dương là **lớp đa số** và được giảm trọng số) |

### `build_neighbor_sampler(graph)`

| | |
|---|---|
| **Nhận** | graph train (NumPy), có self-loop |
| **Trả** | index để lấy mẫu nhanh, **đã bỏ self-loop** |

Mẫu (thật):

| Khoá | Shape | Mẫu | Ý nghĩa |
|---|---|---|---|
| `node_start` | `(126 355,)` | `[0 53 68 100 …]` | node 0 có 53 membership, node 1 có 15 … |
| `node_edges` | `(3 075 381,)` | `[197 21994 22242 …]` | các hyperedge của node 0 |
| `edge_start` | `(274 703,)` | `[0 231 647 …]` | hyperedge 0 có 231 thành viên |
| `edge_nodes` | `(3 075 381,)` | `[26068 26069 …]` | thành viên theo hyperedge |
| `anchor_pool` | `(126 354,)` | `[0 1 2 …]` | node có ít nhất một hyperedge |

### `sample_hyperedge_neighbors(sampler, anchors, count, rng)`

| | |
|---|---|
| **Nhận** | sampler, `anchors (B,)`, `count` |
| **Trả** | `(B, count)`: với mỗi ô, chọn ngẫu nhiên 1 hyperedge của anchor rồi chọn 1 thành viên của nó |
| **Mẫu** | anchors `[64670 59788 95418]`, count 5 → `[[21838 114435 44837 4804 121000], [76452 59927 10517 59788 59688], [95418 96090 95217 94433 95401]]` |

Hàng 2 và hàng 3 có chứa chính anchor (`59788`, `95418`); những ô đó bị che đi
trong loss.

### `contrastive_loss(z0, z_star, anchors, neighbors, temperature)`

| | |
|---|---|
| **Nhận** | `z0`, `z_star (N, 128)`, `anchors (B,)`, `neighbors (B, K)`, `τ = 0.07` |
| **Trả** | scalar: InfoNCE hai chiều, lấy trung bình |
| **Mẫu** | 3 anchor, 4 neighbor trên local graph → `1.8139` |

### `total_loss(output, labels, positive_weight, anchors, neighbors, *, lambda_cl, temperature)`

| | |
|---|---|
| **Nhận** | output của model, nhãn, `pos_weight`, anchors, neighbors, `λ`, `τ` |
| **Trả** | `(loss_tensor, {"bce": float, "contrastive": float})` |
| **Mẫu** | `(0.8342, {"bce": 0.6528, "contrastive": 1.8139})`, trong đó `0.6528 + 0.1 × 1.8139 = 0.8342` |

---

## 8. `8_train.py`

### Kết thúc file thu được

```text
outputs/runs/<run_name>.pt                 checkpoint tốt nhất theo validation AUC
outputs/reports/<run_name>_train.json      lịch sử train + validation tốt nhất
outputs/reports/<run_name>_test.json       kết quả test của checkpoint đó
```

`run_name` ví dụ: `hsl_full_seed_1`.

Nội dung **(minh hoạ; cấu trúc lấy từ code, số liệu chưa phải kết quả thật)**:

```jsonc
// <run>.pt  (torch.save)
{"state_dict": {...}, "input_dim": 92, "settings": {...DEFAULT_SETTINGS...},
 "seed": 1, "epoch": 45, "validation": {"auc": 0.87, "auprc": 0.95, "f1": 0.88, "precision": 0.86, "recall": 0.90}}

// <run>_train.json
{"checkpoint": ".../outputs/runs/hsl_full_seed_1.pt", "best_epoch": 45,
 "best_validation": {"auc": ..., "auprc": ..., "f1": ..., "precision": ..., "recall": ...},
 "settings": {...}, "seed": 1,
 "history": [
   {"epoch": 1, "loss": 0.83, "bce": 0.65, "contrastive": 1.81, "train_auc": 0.61,
    "kept_course": 0.95, "kept_object": 0.95, "kept_behavioral": 0.95, "added": 252708},
   {"epoch": 5, ..., "validation": {"auc": ..., ...}},
   ...]}

// <run>_test.json
{"checkpoint": "...", "checkpoint_epoch": 45, "checkpoint_validation": {...},
 "test": {"auc": ..., "auprc": ..., "f1": ..., "precision": ..., "recall": ...}}
```

`scripts/run_all.sh` gom các file này vào `result/<dd-mm-yyyy_HH-MM>/results.csv`.

### `log(scope, message)`

| | |
|---|---|
| **Nhận** | nhãn và nội dung |
| **Trả** | `None`; in ra `[21:44:33][validation] 16 targets in 0s: auc=0.8929, …` |

### `set_seed(seed)`

| | |
|---|---|
| **Nhận** | seed (`1`, `11`, …) |
| **Trả** | `None`; cố định seed cho `random`, `numpy`, `torch`, CUDA |

### `classification_metrics(labels, probabilities)`

| | |
|---|---|
| **Nhận** | nhãn 0/1, xác suất dropout |
| **Trả** | `{"auc", "auprc", "f1", "precision", "recall"}`; F1, precision, recall tính với ngưỡng 0,5 |
| **Mẫu (toy)** | nhãn `[1,0,1,1,0]`, xác suất `[0.9,0.2,0.4,0.7,0.6]` → `{"auc": 0.8333, "auprc": 0.9167, "f1": 0.6667, "precision": 0.6667, "recall": 0.6667}` |

### `make_run_name(settings, seed)`

| | |
|---|---|
| **Nhận** | settings, seed |
| **Trả** | tên run, dùng làm tên file |
| **Mẫu** | mặc định, seed 1 → `hsl_full_seed_1`; `node_sampling=False, lambda_cl=0`, seed 11 → `hsl_no-node_no-cl_full_seed_11`; `hsl=False`, seed 111 → `hgnn_full_seed_111` |

### `make_model(input_dim, settings)`

| | |
|---|---|
| **Nhận** | số feature (`92`, hoặc ít hơn nếu dùng feature set khác), settings |
| **Trả** | `HSLModel` (cấu trúc ở mục 5); `hsl=False` → không có `structure_learner` |

### `evaluate(model, split_data, settings, device, *, limit, seed)`

| | |
|---|---|
| **Nhận** | model, kết quả `load_evaluation_split`, `limit` (0 = mọi target) |
| **Làm** | dựng local graph theo batch `eval_batch_size = 8` → forward → lấy logit của target |
| **Trả** | dict metric như `classification_metrics` |
| **Mẫu** | model **chưa train**, 16 target validation → `{"auc": 0.8929, "auprc": 0.9846, "f1": 0.0, "precision": 0.0, "recall": 0.0}` |

F1 bằng 0 vì model chưa train cho mọi xác suất ≈ 0,47, dưới ngưỡng 0,5. Riêng
AUC cao là do ngẫu nhiên trên 16 mẫu, không mang ý nghĩa.

### `train(settings, *, seed, output_dir, device_name)`

| | |
|---|---|
| **Nhận** | settings, seed |
| **Làm** | mỗi epoch: forward cả train graph → `total_loss` → backward → mỗi `eval_every = 5` epoch thì validate; lưu checkpoint khi AUC tăng; dừng sau `patience = 5` lần validate không cải thiện |
| **Trả** | dict report (giống `<run>_train.json`) |

### `test(checkpoint_path, *, output_dir, device_name, limit)`

| | |
|---|---|
| **Nhận** | đường dẫn `.pt` |
| **Trả** | dict report (giống `<run>_test.json`) |

---

## Ghi chú rút ra khi lấy mẫu

1. **Bước 2 không tự chạy lại** nếu đã có ba file CSV. Xoá chúng nếu muốn tạo lại.
2. **Cột `close_info` bằng hằng số** trên train (std = 0), nên cột này không mang
   thông tin.
3. **71,6% node thiếu tuổi.** Cột `age_at_course_start` chủ yếu là giá trị median
   được điền vào; cột `age_missing` mang nhiều thông tin hơn.
4. **Dropout là lớp đa số (75,8%)**, nên `pos_weight = 0,319`, tức BCE giảm trọng
   số lớp dropout. Khi báo F1, cần nói rõ lớp dương là dropout.
5. **Local graph khá lớn:** target 0 của validation kéo theo 753 node train (vì
   course có 734 thành viên). Đây là lý do evaluate chậm hơn một epoch train.
