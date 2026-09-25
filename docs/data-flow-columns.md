# Dòng chảy dữ liệu và các cột qua từng giai đoạn

Tài liệu này trả lời bốn câu hỏi:

1. Dataset ban đầu có những bảng và cột nào được project sử dụng?
2. Một raw event biến thành một dòng processed CSV như thế nào?
3. Một enrollment biến thành vector 94 chiều trong `X.npy` như thế nào?
4. Từ `X.npy`, dữ liệu được đưa vào hypergraph và mô hình ra sao?

Các dòng dữ liệu bên dưới là **ví dụ minh họa nhất quán**, không phải dòng thật
của dataset. Schema, tên cột và quy tắc biến đổi bám theo phiên bản code hiện tại.

---

## 1. Bức tranh tổng thể

```text
Ba file tải xuống
│
├── user_info.csv
├── course_info.csv
└── prediction_data.tar.gz
    ├── train_log.csv
    ├── train_truth.csv
    ├── test_log.csv
    └── test_truth.csv
             │
             ▼
       2_preprocess.py
             │
             ├── train.csv
             ├── validation.csv
             └── test.csv
                    │
                    ▼
              3_features.py
                    │
                    ├── train/X.npy
                    ├── validation/X.npy
                    ├── test/X.npy
                    └── feature_names.csv
                           │
                           ▼
                     4_hypergraph.py
                           │
                           └── hypergraph.npz
                                  │
                                  ▼
                             8_train.py
                                  │
                                  ├── checkpoint .pt
                                  ├── train report .json
                                  └── test report .json
```

Đơn vị dữ liệu thay đổi qua pipeline:

| Giai đoạn | Một dòng hoặc một row đại diện cho |
|---|---|
| Raw log | Một hành động của một enrollment |
| Processed CSV | Một hành động đã ghép user, course và label |
| `X.npy` | Một enrollment, tức một node |
| `hypergraph.npz` | Sparse H0, edge metadata, evaluation neighbors và kNN settings |
| Model output | Một dropout logit cho mỗi node được dự đoán |

---

## 2. Giai đoạn tải xuống: ba raw file

Sau `1_download.py`:

```text
data/raw/xuetangx/
├── prediction_data.tar.gz
├── user_info.csv
└── course_info.csv
```

`prediction_data.tar.gz` được đọc trực tiếp, không cần giải nén ra thư mục.

### 2.1 `user_info.csv`

Các cột được code sử dụng:

| Cột | Kiểu sau khi đọc | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `user_id` | `int` | ID người học | `10` |
| `gender` | `str` | Giới tính | `female` |
| `education` | `str` | Trình độ học vấn | `Bachelor's` |
| `birth` | `str`, sau đó đổi sang số khi hợp lệ | Năm sinh | `1995` |

Ví dụ:

```csv
user_id,gender,education,birth
10,female,Bachelor's,1995
11,male,Master's,1990
12,,High,no data
```

Ở bước này missing value vẫn là chuỗi raw. Nó chỉ được diễn giải ở
`3_features.py`.

### 2.2 `course_info.csv`

| Cột | Kiểu sau khi đọc | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `course_id` | `str` | ID khóa học | `course-A` |
| `start` | `str` ISO date/time | Ngày bắt đầu khóa học | `2024-01-01` |
| `end` | `str` ISO date/time | Ngày kết thúc | `2024-03-01` |
| `category` | `str` | Lĩnh vực khóa học | `computer` |

Ví dụ:

```csv
course_id,start,end,category
course-A,2024-01-01,2024-03-01,computer
course-B,2024-02-01,2024-04-01,math
```

`start` là bắt buộc đối với code hiện tại vì nó được dùng để tính
`course_day`. `end` có thể thiếu; khi đó duration sẽ được impute ở bước feature.

### 2.3 `train_truth.csv` và `test_truth.csv` trong archive

| Cột | Kiểu | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `enroll_id` | `int` | ID enrollment | `100` |
| `truth` | `int` | Label dropout | `0` hoặc `1` |

Ví dụ:

```csv
enroll_id,truth
100,0
101,1
```

Trong project:

```text
truth = 1 → dropout
truth = 0 → không dropout
```

`train_truth.csv` cung cấp enrollment để chia train/validation.
`test_truth.csv` chỉ thuộc official test.

### 2.4 `train_log.csv` và `test_log.csv` trong archive

Sáu cột mà code hiện tại trực tiếp sử dụng:

| Cột raw | Cách dùng | Ví dụ |
|---|---|---|
| `enroll_id` | Nối event với label và split | `100` |
| `username` | Được đổi thành `user_id` | `10` |
| `course_id` | Nối với `course_info.csv` | `course-A` |
| `action` | Tạo behavior feature và object type | `play_video` |
| `object` | Được đổi thành `object_id` | `video-01` |
| `time` | Tính ngày hoạt động kể từ course start | `2024-01-02T08:00:00` |

Ví dụ raw log của enrollment `100`:

```csv
enroll_id,username,course_id,action,object,time
100,10,course-A,play_video,video-01,2024-01-02T08:00:00
100,10,course-A,pause_video,video-01,2024-01-02T08:05:00
100,10,course-A,problem_check,problem-07,2024-01-04T10:00:00
```

Nếu raw file có thêm cột, `csv.DictReader` vẫn đọc được nhưng pipeline hiện tại
không sử dụng cột đó.

---

## 3. Giai đoạn preprocess: ghép dữ liệu và chia split

### 3.1 Quy tắc chia dữ liệu

```text
train_truth enrollment IDs
        │
        ├── shuffle bằng SPLIT_SEED = 1
        ├── 80% đầu → train
        └── 20% còn lại → validation

test_truth enrollment IDs
        └── giữ nguyên → test
```

`test_log.csv` không được trộn vào train hoặc validation.

Mỗi split có hệ `node_id` cục bộ riêng:

```text
train node_id:      0, 1, 2, ...
validation node_id: 0, 1, 2, ...
test node_id:       0, 1, 2, ...
```

### 3.2 Cách tính `course_day`

Với event:

```text
event date   = 2024-01-02
course start = 2024-01-01
```

Ta có:

```text
course_day = event date - course start = 1
```

Chỉ giữ:

```text
0 <= course_day < 35
```

Vì vậy pipeline quan sát ngày `0` đến ngày `34`.

### 3.3 Schema chung của `train.csv`, `validation.csv`, `test.csv`

Ba file có cùng 14 cột:

| # | Cột | Nguồn hoặc cách sinh | Ví dụ |
|---:|---|---|---|
| 1 | `node_id` | Sinh mới, cục bộ trong split | `0` |
| 2 | `enroll_id` | Raw log/truth | `100` |
| 3 | `user_id` | Raw `username` | `10` |
| 4 | `course_id` | Raw log | `course-A` |
| 5 | `label` | `truth` theo enrollment | `0` |
| 6 | `gender` | Nối từ `user_info.csv` | `female` |
| 7 | `education` | Nối từ `user_info.csv` | `Bachelor's` |
| 8 | `birth` | Nối từ `user_info.csv` | `1995` |
| 9 | `course_start` | Nối từ `course_info.csv:start` | `2024-01-01` |
| 10 | `course_end` | Nối từ `course_info.csv:end` | `2024-03-01` |
| 11 | `category` | Nối từ `course_info.csv` | `computer` |
| 12 | `action` | Raw log | `play_video` |
| 13 | `object_id` | Raw `object` | `video-01` |
| 14 | `course_day` | Tính từ event date và course start | `1` |

### 3.4 Ví dụ sau khi ghép

Giả sử enrollment `100` được chia vào train và được gán `node_id=0`:

```csv
node_id,enroll_id,user_id,course_id,label,gender,education,birth,course_start,course_end,category,action,object_id,course_day
0,100,10,course-A,0,female,Bachelor's,1995,2024-01-01,2024-03-01,computer,play_video,video-01,1
0,100,10,course-A,0,female,Bachelor's,1995,2024-01-01,2024-03-01,computer,pause_video,video-01,1
0,100,10,course-A,0,female,Bachelor's,1995,2024-01-01,2024-03-01,computer,problem_check,problem-07,3
```

Điểm quan trọng:

- Ba dòng vẫn chỉ là **một node** vì đều có `node_id=0`.
- Thông tin node được lặp lại để file tự chứa đủ dữ liệu.
- Ba event sẽ được cộng dồn khi tạo feature.
- Label `0` cũng được lặp lại, nhưng khi train chỉ lấy một label cho node 0.

### 3.5 Enrollment không có event hợp lệ

Nếu enrollment tồn tại nhưng mọi event đều ngoài 35 ngày hoặc action không nằm
trong `ACTIONS`, code vẫn ghi một dòng:

```csv
node_id,enroll_id,user_id,course_id,label,gender,education,birth,course_start,course_end,category,action,object_id,course_day
1,101,11,course-B,1,male,Master's,1990,2024-02-01,2024-04-01,math,,,
```

Ba cột cuối rỗng. Kết quả:

- Node và label không bị mất.
- 58 behavior counts của node đều bằng `0` trước chuẩn hóa.
- User/course feature vẫn được tạo bình thường.

---

## 4. Giai đoạn feature: từ nhiều event rows thành một node row

### 4.1 Các array trung gian trong `build_split_features()`

Với một split có `N` node:

| Biến | Shape | Kiểu | Nội dung trước chuẩn hóa |
|---|---|---|---|
| `behavior_features` | `(N, 58)` | `int32` | 35 day counts + 23 action counts |
| `user_context` | `(N, 15)` | `float32` | Gender, education, age, missing flag |
| `course_context` | `(N, 21)` | `float32` | Category, duration, missing flag |
| `ages` | `(N,)` | `float64` | Tuổi thô hoặc `NaN` |
| `durations` | `(N,)` | `float64` | Số ngày thô hoặc `NaN` |

Các array này chỉ tồn tại trong RAM, không được ghi thành file riêng.

### 4.2 Ví dụ cộng behavior cho node 0

Từ ba processed rows ở trên:

```text
Ngày 1: play_video, pause_video
Ngày 3: problem_check
```

Behavior count thô:

```text
activity_day_1       = 2
activity_day_3       = 1
action_video_2       = 1   # play_video
action_video_3       = 1   # pause_video
action_assignment_2  = 1   # problem_check
các behavior khác    = 0
```

Mỗi event làm tăng đúng hai vị trí:

```text
một activity_day column + một action column
```

### 4.3 Ví dụ tạo user context

Raw context:

```text
gender       = female
education    = Bachelor's
birth        = 1995
course_start = 2024-01-01
```

Trước khi scale age:

```text
gender_female       = 1
gender_male         = 0
gender_missing      = 0
gender_other        = 0
education_bachelors = 1
các education khác  = 0
age                 = 2024 - 1995 = 29
age_missing         = 0
```

### 4.4 Ví dụ tạo course context

Raw context:

```text
category     = computer
course_start = 2024-01-01
course_end   = 2024-03-01
```

Năm 2024 là năm nhuận nên ví dụ này có:

```text
category_computer       = 1
các category khác       = 0
course_duration_days    = 60
course_duration_missing = 0
```

### 4.5 Missing và other

Các chuỗi sau được xem là missing, không phân biệt hoa thường sau `.lower()`:

```text
"", "na", "n/a", "none", "null", "no data", "-"
```

Ví dụ:

| Raw value | Feature nhận 1 |
|---|---|
| `gender=""` | `gender_missing` |
| `gender="unknown"` | `gender_other` |
| `education="n/a"` | `education_missing` |
| `education="College"` | `education_other` |
| `category="null"` | `category_missing` |
| `category="law"` | `category_other` |

Birth thuộc danh sách missing hoặc tạo age ngoài `[10, 100]` làm `age=NaN`.
Course end thuộc danh sách missing hoặc sớm hơn course start làm `duration=NaN`.
Hai giá trị này được xử lý ở bước kế tiếp. Code hiện tại giả định các giá trị
không thiếu vẫn đúng định dạng số/ngày; chuỗi sai định dạng có thể làm chương
trình dừng thay vì tự sửa.

---

## 5. Fit trên train, transform train/validation/test

### 5.1 Behavior

Với mỗi behavior column:

```text
raw count
   ↓ log1p
log(1 + count)
   ↓ chuẩn hóa bằng train mean/std
(value - train_mean) / train_std
```

Ví dụ minh họa cho `activity_day_1`:

```text
node 0 raw count = 2
log1p(2)          = 1.0986

giả sử train mean = 0.50
giả sử train std  = 0.40

X[0, activity_day_1] = (1.0986 - 0.50) / 0.40
                     ≈ 1.4965
```

`0.50` và `0.40` chỉ là số minh họa. Code tính thống kê thật từ toàn bộ train.

### 5.2 Age và duration

Train được dùng để fit ba giá trị:

```text
median, mean, standard deviation
```

Sau đó cả ba split đều dùng lại thống kê train:

```text
missing → điền train median
filled value → (value - train mean) / train std
missing flag → 1 nếu raw value ban đầu bị thiếu
```

### 5.3 Validation được xử lý thế nào?

Ví dụ validation có `activity_day_1=1`:

```text
log1p(1) = 0.6931

validation value = (0.6931 - 0.50) / 0.40
                 ≈ 0.4828
```

Validation **không tính mean/std riêng**. Test cũng vậy.

| Split | Có fit statistics không? | Statistics sử dụng |
|---|---:|---|
| Train | Có | Train |
| Validation | Không | Train |
| Test | Không | Train |

One-hot columns không qua mean/std nên vẫn là `0` hoặc `1`.

---

## 6. Toàn bộ 94 columns trong `X.npy`

`X.npy` không có header. Ý nghĩa từng index được lưu trong `feature_names.csv`.

### 6.1 Behavior theo ngày: index 0–34

| Index | Feature | Nguồn |
|---:|---|---|
| `0` | `activity_day_0` | Số event có `course_day=0` |
| `1` | `activity_day_1` | Số event có `course_day=1` |
| `2` | `activity_day_2` | Số event có `course_day=2` |
| `3` | `activity_day_3` | Số event có `course_day=3` |
| `4` | `activity_day_4` | Số event có `course_day=4` |
| `5` | `activity_day_5` | Số event có `course_day=5` |
| `6` | `activity_day_6` | Số event có `course_day=6` |
| `7` | `activity_day_7` | Số event có `course_day=7` |
| `8` | `activity_day_8` | Số event có `course_day=8` |
| `9` | `activity_day_9` | Số event có `course_day=9` |
| `10` | `activity_day_10` | Số event có `course_day=10` |
| `11` | `activity_day_11` | Số event có `course_day=11` |
| `12` | `activity_day_12` | Số event có `course_day=12` |
| `13` | `activity_day_13` | Số event có `course_day=13` |
| `14` | `activity_day_14` | Số event có `course_day=14` |
| `15` | `activity_day_15` | Số event có `course_day=15` |
| `16` | `activity_day_16` | Số event có `course_day=16` |
| `17` | `activity_day_17` | Số event có `course_day=17` |
| `18` | `activity_day_18` | Số event có `course_day=18` |
| `19` | `activity_day_19` | Số event có `course_day=19` |
| `20` | `activity_day_20` | Số event có `course_day=20` |
| `21` | `activity_day_21` | Số event có `course_day=21` |
| `22` | `activity_day_22` | Số event có `course_day=22` |
| `23` | `activity_day_23` | Số event có `course_day=23` |
| `24` | `activity_day_24` | Số event có `course_day=24` |
| `25` | `activity_day_25` | Số event có `course_day=25` |
| `26` | `activity_day_26` | Số event có `course_day=26` |
| `27` | `activity_day_27` | Số event có `course_day=27` |
| `28` | `activity_day_28` | Số event có `course_day=28` |
| `29` | `activity_day_29` | Số event có `course_day=29` |
| `30` | `activity_day_30` | Số event có `course_day=30` |
| `31` | `activity_day_31` | Số event có `course_day=31` |
| `32` | `activity_day_32` | Số event có `course_day=32` |
| `33` | `activity_day_33` | Số event có `course_day=33` |
| `34` | `activity_day_34` | Số event có `course_day=34` |

Tất cả 35 cột này được `log1p` rồi chuẩn hóa bằng train statistics.

### 6.2 Behavior theo action: index 35–57

| Index | Feature name | Raw action |
|---:|---|---|
| 35 | `action_video_1` | `seek_video` |
| 36 | `action_video_2` | `play_video` |
| 37 | `action_video_3` | `pause_video` |
| 38 | `action_video_4` | `stop_video` |
| 39 | `action_video_5` | `load_video` |
| 40 | `action_assignment_1` | `problem_get` |
| 41 | `action_assignment_2` | `problem_check` |
| 42 | `action_assignment_3` | `problem_save` |
| 43 | `action_assignment_4` | `reset_problem` |
| 44 | `action_assignment_5` | `problem_check_correct` |
| 45 | `action_assignment_6` | `problem_check_incorrect` |
| 46 | `action_forum_1` | `create_thread` |
| 47 | `action_forum_2` | `create_comment` |
| 48 | `action_forum_3` | `delete_thread` |
| 49 | `action_forum_4` | `delete_comment` |
| 50 | `action_forum_5` | `close_forum` |
| 51 | `action_web_page_1` | `click_info` |
| 52 | `action_web_page_2` | `click_courseware` |
| 53 | `action_web_page_3` | `click_about` |
| 54 | `action_web_page_4` | `click_forum` |
| 55 | `action_web_page_5` | `click_progress` |
| 56 | `action_web_page_6` | `close_courseware` |
| 57 | `action_web_page_7` | `close_info` |

Tất cả 23 cột này cũng được `log1p` rồi chuẩn hóa bằng train statistics.

### 6.3 User context: index 58–72

| Index | Feature name | Giá trị |
|---:|---|---|
| 58 | `gender_female` | 1 nếu gender là `female` |
| 59 | `gender_male` | 1 nếu gender là `male` |
| 60 | `gender_missing` | 1 nếu gender thiếu |
| 61 | `gender_other` | 1 nếu gender ngoài vocabulary |
| 62 | `education_associate` | 1 nếu `Associate` |
| 63 | `education_bachelors` | 1 nếu `Bachelor's` |
| 64 | `education_doctorate` | 1 nếu `Doctorate` |
| 65 | `education_high` | 1 nếu `High` |
| 66 | `education_masters` | 1 nếu `Master's` |
| 67 | `education_middle` | 1 nếu `Middle` |
| 68 | `education_primary` | 1 nếu `Primary` |
| 69 | `education_missing` | 1 nếu education thiếu |
| 70 | `education_other` | 1 nếu education ngoài vocabulary |
| 71 | `age_at_course_start` | Age sau impute và standardize |
| 72 | `age_missing` | 1 nếu birth thiếu hoặc age không hợp lệ |

### 6.4 Course context: index 73–93

| Index | Feature name | Giá trị |
|---:|---|---|
| 73 | `category_art` | 1 nếu category là `art` |
| 74 | `category_biology` | 1 nếu `biology` |
| 75 | `category_business` | 1 nếu `business` |
| 76 | `category_chemistry` | 1 nếu `chemistry` |
| 77 | `category_computer` | 1 nếu `computer` |
| 78 | `category_economics` | 1 nếu `economics` |
| 79 | `category_education` | 1 nếu `education` |
| 80 | `category_electrical` | 1 nếu `electrical` |
| 81 | `category_engineering` | 1 nếu `engineering` |
| 82 | `category_foreign_language` | 1 nếu `foreign language` |
| 83 | `category_history` | 1 nếu `history` |
| 84 | `category_literature` | 1 nếu `literature` |
| 85 | `category_math` | 1 nếu `math` |
| 86 | `category_medicine` | 1 nếu `medicine` |
| 87 | `category_philosophy` | 1 nếu `philosophy` |
| 88 | `category_physics` | 1 nếu `physics` |
| 89 | `category_social_science` | 1 nếu `social science` |
| 90 | `category_missing` | 1 nếu category thiếu |
| 91 | `category_other` | 1 nếu category ngoài vocabulary |
| 92 | `course_duration_days` | Duration sau impute và standardize |
| 93 | `course_duration_missing` | 1 nếu end date thiếu/không hợp lệ |

Tổng số cột:

```text
35 day + 23 action + 15 user + 21 course = 94
```

### 6.5 Ví dụ một row trong `X.npy`

`X.npy` là NumPy array:

```python
X.shape == (node_count, 94)
X.dtype == np.float32
```

Row `X[0]` tương ứng trực tiếp với `node_id=0`.

Một số giá trị minh họa:

```text
X[0, 1]  =  1.4965  # activity_day_1 sau log + scale
X[0, 3]  =  0.7320  # activity_day_3 sau log + scale
X[0, 36] =  0.4180  # play_video sau log + scale
X[0, 37] =  0.5270  # pause_video sau log + scale
X[0, 41] =  0.3050  # problem_check sau log + scale
X[0, 58] =  1.0000  # gender_female
X[0, 63] =  1.0000  # education_bachelors
X[0, 71] = -0.6000  # age 29 sau scale, giá trị minh họa
X[0, 77] =  1.0000  # category_computer
X[0, 92] =  0.5000  # duration 60 sau scale, giá trị minh họa
```

Các số standardized thực tế phụ thuộc thống kê train thật.

### 6.6 `feature_names.csv`

Schema:

```csv
feature_index,feature_name,source
```

Ví dụ:

```csv
feature_index,feature_name,source
0,activity_day_0,course_day=0
1,activity_day_1,course_day=1
36,action_video_2,play_video
58,gender_female,gender=female
71,age_at_course_start,course start year - birth year
77,category_computer,category=computer
92,course_duration_days,course end - course start
```

File này là “header bên ngoài” của `X.npy`.

---

## 7. Feature-set dùng trong thí nghiệm

| `feature_set` | Columns | Số chiều |
|---|---|---:|
| `behavior` | 0–57 | 58 |
| `behavior_user` | 0–72 | 73 |
| `behavior_course` | 0–57 và 73–93 | 79 |
| `full` | 0–93 | 94 |

Ví dụ:

```python
feature_set = "behavior"
```

Model không nhìn thấy gender, education, age, category hoặc duration; nó chỉ
nhận 58 behavior columns.

---

## 8. Giai đoạn hypergraph

Ba họ hyperedge được tạo như sau:

| Family | Khóa gom nhóm | Dữ liệu sử dụng |
|---|---|---|
| `course` | `course_id` | Các train node học cùng khóa |
| `object` | `(course_id, object_id, object_type)` | Chỉ video, assignment, forum có object hợp lệ |
| `behavioral` | Anchor node + các behavioral neighbors | Cosine similarity trên 58 behavior columns |

Web-page action vẫn có behavior feature nhưng không tạo Object hyperedge.

### 8.1 Exact behavioral neighbors

Mỗi row lưu ID các train node gần nhất theo exact cosine similarity của 58
behavior features. Tính toán dùng PyTorch theo query batch, nên không tạo toàn bộ
ma trận similarity `N × N` trong bộ nhớ.

Ví dụ với `k_max=3`:

```python
train_neighbors = [
    [2, 5, 1],  # ba train neighbors của train node 0
    [0, 3, 4],  # ba train neighbors của train node 1
]
```

Đối với validation/test, giá trị trong row luôn là **train node ID**:

```python
validation_neighbors[0] = [2, 5, 1]
```

Nghĩa là validation node 0 giống các train node 2, 5 và 1 nhất.

### 8.2 Thứ tự hyperedge

Edge ID được cấp theo ba block liên tục và cố định:

1. Tất cả Course hyperedge, sort theo `course_id`.
2. Tất cả Object hyperedge, sort theo course, object type và object ID.
3. Một Behavioral kNN hyperedge cho mỗi train anchor.

Hai anchor có cùng member set vẫn giữ hai behavioral hyperedge riêng. Với `k=10`,
mỗi behavioral edge gồm anchor và tối đa 10 train neighbors.

### 8.3 Sparse H0

`H0` là sparse CSR incidence matrix:

```text
rows    = train node IDs
columns = edge IDs
value   = 1 nếu node thuộc edge, ngược lại 0
```

Ví dụ một phần `H0`:

```text
          edge 0  edge 1  edge 2
node 0       1       1       1
node 1       1       0       1
node 2       1       1       0
node 3       0       0       1
node 5       0       0       1
```

Shape:

```python
H0.shape == (train_node_count, hyperedge_count)
```

Validation/test không được thêm thành row trong train `H0`. Mỗi target được dựng
một local graph với train nodes làm reference.

### 8.4 `hypergraph.npz`

Một bundle duy nhất chứa:

| Array/scalar | Ý nghĩa |
|---|---|
| `h0_data`, `h0_indices`, `h0_indptr`, `h0_shape` | Thành phần CSR của H0 |
| `edge_families`, `edge_sizes` | Metadata thẳng hàng với cột H0 |
| `train_neighbors` | `k_max` train-neighbor IDs cho mỗi train node |
| `validation_neighbors`, `test_neighbors` | `k_max` train-reference neighbor IDs |
| `k`, `k_max` | Kích thước neighbor đang dùng và kích thước đã lưu |
| `neighbor_backend` | `torch_exact_cosine` |

`k_max` quyết định số neighbor được tính và lưu; `k` quyết định số neighbor đầu
tiên thực sự tham gia Behavioral hyperedge.

---

## 9. Dữ liệu đi vào model và loss

Khi train, `load_train_graph()` trả:

| Biến | Shape ví dụ | Nguồn |
|---|---|---|
| `train_node_features` | `(N, F)` | `train/X.npy` sau chọn feature-set |
| `initial_incidence_matrix` | `(N, E)` | CSR arrays trong `hypergraph.npz` |
| `labels` | `(N,)` | Một label cho mỗi node trong `train.csv` |
| `families` | `(E,)` | `hypergraph.npz:edge_families` |
| `sizes` | `(E,)` | `hypergraph.npz:edge_sizes` |

Với `feature_set="full"`:

```text
F = 94
```

Với `feature_set="behavior"`:

```text
F = 58
```

Model tạo một raw dropout logit cho mỗi node:

```text
logits shape = (N,)
probability  = sigmoid(logit)
```

Ví dụ:

```text
node_id = 0
label   = 0
logit   = -1.20
score   = sigmoid(-1.20) ≈ 0.231
```

`score` càng gần 1 thì mô hình càng nghiêng về lớp dropout.

---

## 10. Checkpoint và report

### 10.1 Checkpoint `.pt`

Các key chính:

```python
{
    "state_dict": ...,   # trọng số model
    "epoch": ...,        # epoch có validation AUC tốt nhất
    "validation": ...,   # metrics tại epoch đó
    "input_dim": ...,    # 58, 73, 79 hoặc 94
    "settings": ...,     # seed, feature-set, learning rate, ...
}
```

### 10.2 Training report `.json`

Mỗi history record có dạng:

```json
{
  "epoch": 1,
  "loss": 0.812,
  "bce": 0.701,
  "contrastive": 1.110,
  "validation": {
    "auc": 0.78,
    "auprc": 0.76,
    "precision": 0.69,
    "recall": 0.75,
    "f1": 0.72
  }
}
```

Tên key trong ví dụ đúng theo code hiện tại; các giá trị chỉ minh họa luồng.

### 10.3 Test report `.json`

Test report liên kết kết quả cuối với checkpoint đã chọn bằng validation:

```json
{
  "checkpoint": "outputs/runs/simple_hgsl_full_seed_1.pt",
  "checkpoint_epoch": 12,
  "checkpoint_validation": {"auc": 0.81},
  "test_targets": 67000,
  "test_metrics": {"auc": 0.80}
}
```

Test không tham gia chọn checkpoint.

---

## 11. Theo dõi một enrollment từ đầu đến cuối

### Raw

```text
enroll_id=100
user_id=10
course_id=course-A
label=0
3 event rows
```

### Processed

```text
train.csv có 3 rows với node_id=0
```

### Feature

```text
3 event rows được aggregate thành X[0]
X[0] có 94 columns
```

### Hypergraph

```text
node 0 tham gia:
- một Course edge của course-A
- các Object edge như video-01, problem-07 nếu có node khác cùng tương tác
- một Behavioral edge với các train node gần nhất
```

### Model

```text
X[0] + hypergraph connections
        ↓
HGNN/HSL
        ↓
dropout logit cho node 0
        ↓ sigmoid
dropout probability
```

---

## 12. Những điểm cần nhớ

1. Processed CSV có một dòng trên event; `X.npy` có một row trên node.
2. `node_id` chính là row index trong `X.npy` của cùng split.
3. Label nằm trong split CSV, không nằm trong `X.npy`.
4. `feature_names.csv` ánh xạ index sang ý nghĩa của 94 columns.
5. Chỉ train được dùng để fit median, mean và std.
6. Validation/test dùng train statistics nhưng giữ feature values của chính nó.
7. Train `H0` chỉ chứa train nodes.
8. Validation/test target chỉ nối tới train reference nodes khi đánh giá.
9. Official test không được dùng để chọn checkpoint.
