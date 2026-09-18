# XuetangX Full: cấu trúc dữ liệu, chia tập và đặc trưng

Tài liệu này chỉ sử dụng **một dataset: XuetangX Full**. Các file activity,
train/test log, truth, user và course là các bảng thành phần của cùng dataset,
không phải các phiên bản dataset khác nhau.

Số liệu dưới đây được quét trực tiếp từ dữ liệu đang có trong project.

## 1. Dataset gồm những gì?

| Thành phần | Quy mô đã kiểm tra | Các field | Mục đích |
|---|---:|---|---|
| 6 file `activity_*.csv.gz` | 351.452.376 event | `course_id`, `user_id`, `session_id`, `action`, `time` | Activity toàn nền tảng; chỉ dùng khi thử pretraining |
| `train_log.csv` | 29.165.540 event; 157.943 enrollment | `enroll_id`, `username`, `course_id`, `session_id`, `action`, `object`, `time` | Hành vi của phần train do nguồn cung cấp |
| `test_log.csv` | 12.944.862 event; 67.699 enrollment | Giống `train_log.csv` | Hành vi của phần test do nguồn cung cấp |
| `train_truth.csv` | 157.943 nhãn | `enroll_id`, `truth` | Nhãn train |
| `test_truth.csv` | 67.699 nhãn | `enroll_id`, `truth` | Nhãn test |
| `user_info.csv` | 9.627.148 user | `user_id`, `gender`, `education`, `birth` | Metadata người học |
| `course_info.csv` | 6.410 course | `id`, `course_id`, `start`, `end`, `course_type`, `category` | Metadata khóa học và mốc thời gian |

Lưu ý quan trọng:

- Dataset **có nhãn**, được lưu riêng trong `train_truth.csv` và
  `test_truth.csv`: `1 = dropout`, `0 = non-dropout`.
- Sáu file activity toàn nền tảng không có `enroll_id`, `object` hoặc `truth`.
  Điều này chỉ mô tả schema của bảng activity, không có nghĩa toàn dataset không có nhãn.
- `train` và `test` là hai phần do nguồn cung cấp bên trong XuetangX Full.

## 2. Quy mô phần có nhãn

| Chỉ số | Train nguồn | Test nguồn | Tổng |
|---|---:|---:|---:|
| Event thô | 29.165.540 | 12.944.862 | 42.110.402 |
| Enrollment | 157.943 | 67.699 | 225.642 |
| User khác nhau | 69.823 | 44.008 | 77.083 sau khi gộp |
| Course khác nhau | 247 | 247 | 247 |
| Dropout (`truth=1`) | 119.817 | 51.316 | 171.133 (75,84%) |
| Non-dropout (`truth=0`) | 38.126 | 16.383 | 54.509 (24,16%) |

Có **36.748 user** và toàn bộ **247 course** xuất hiện ở cả train và test nguồn.
Vì vậy, split nguồn phù hợp để đối chiếu baseline nhưng không phù hợp làm kết quả
chính nếu muốn đánh giá trên người học chưa từng thấy.

Sau khi chỉ giữ `course_day = 0..34`:

| Chỉ số | Giá trị |
|---|---:|
| Event được giữ | 40.558.640 |
| Event từ ngày 35 trở đi bị loại | 1.551.762 |
| Enrollment không có event trong cửa sổ 35 ngày | 3.305 |
| Event thiếu `object` trong cửa sổ | 10.596.903 (26,13%) |
| Thiếu `session_id`, `action`, `time` trong cửa sổ | 0 |

Trong 77.083 user có nhãn, số user thiếu `gender`, `education`, `birth` lần lượt
là 48.664, 57.731 và 60.711. Trong 247 course được dùng, 2 course thiếu
`category`; `start`, `end` và `course_type` không thiếu. Cả 247 course đều có
`course_type=0`, nên field này không mang thông tin phân biệt trong bài toán hiện tại.

## 3. Đơn vị dự đoán và cửa sổ quan sát

- Một node là một enrollment, xác định bởi `enroll_id` và cặp
  `(user_id, course_id)`.
- `username` trong log được chuẩn hóa tên thành `user_id`.
- `course_day = date(time) - date(course_info.start)`.
- Chỉ dùng event ở ngày 0–34; timestamp nguồn không công bố múi giờ nên không tự
  chuyển timezone.
- Label được đọc nguyên trạng từ hai file truth; không tự suy nhãn từ activity.

## 4. Chia train, validation và test

### Chia tập chính

Gộp toàn bộ 225.642 enrollment có nhãn, sau đó chia theo **nhóm `user_id`** với
năm seed cố định `1, 11, 111, 1111, 11111`:

| Tập | Tỷ lệ user mục tiêu | Dùng để làm gì |
|---|---:|---|
| Train | 64% | Fit mô hình, scaler, vocabulary, kNN và hypergraph |
| Validation | 16% | Chọn siêu tham số và early stopping |
| Test | 20% | Báo cáo kết quả cuối cùng |

Mỗi seed có cùng quy mô và tỷ lệ nhãn sau:

| Tập | User | Enrollment | Dropout | Dropout rate |
|---|---:|---:|---:|---:|
| Train | 49.333 | 144.543 | 109.625 | 75,84% |
| Validation | 12.333 | 36.028 | 27.374 | 75,98% |
| Test | 15.417 | 45.071 | 34.134 | 75,73% |

Thành viên của từng tập thay đổi theo seed; kết quả cuối báo cáo mean ± std trên
năm seed.

Ràng buộc bắt buộc:

- Một user chỉ thuộc một tập; user overlap giữa ba tập phải bằng 0.
- Course được phép xuất hiện ở nhiều tập vì mục tiêu chính là dự đoán cho người học mới.
- Tỷ lệ dropout cần được cân bằng gần nhau giữa các tập; số enrollment thực tế có
  thể lệch nhẹ so với tỷ lệ user.
- Mọi phép học tham số tiền xử lý chỉ fit trên train rồi áp dụng sang validation/test.
- Split train/test do nguồn cung cấp chỉ giữ cho thí nghiệm đối chiếu baseline.

### Activity toàn nền tảng

Phần activity có 351.452.376 event, 772.887 user, 1.629 course và 2.940.490
cặp user–course. Nó không dùng trong thí nghiệm chính ở vòng đầu. Nếu thử
pretraining, phải loại toàn bộ user thuộc validation và test trước khi học để tránh
rò rỉ thông tin.

## 5. Action vocabulary

Giữ vocabulary cố định gồm 23 action:

| Nhóm | Action |
|---|---|
| Video | `seek_video`, `play_video`, `pause_video`, `stop_video`, `load_video` |
| Bài tập | `problem_get`, `problem_check`, `problem_save`, `reset_problem`, `problem_check_correct`, `problem_check_incorrect` |
| Diễn đàn | `create_thread`, `create_comment`, `delete_thread`, `delete_comment`, `close_forum` |
| Trang web | `click_info`, `click_courseware`, `click_about`, `click_forum`, `click_progress`, `close_courseware`, `close_info` |

Phần có nhãn quan sát 22 action; `close_info` có count bằng 0. Vẫn giữ cột này để
schema thống nhất với activity toàn nền tảng.

## 6. Các field tự sinh

| Field | Cách sinh |
|---|---|
| `node_id` | Đánh số liên tục cho enrollment sau khi khóa tập dữ liệu |
| `course_day` | `date(time) - date(course_start)`; chỉ giữ 0–34 |
| `object_key` | `(course_id, object)`; không dùng `object` riêng lẻ |
| `object_type` | Suy từ nhóm action: video, bài tập, diễn đàn hoặc trang web |
| `day_00..day_34` | Số event của enrollment theo từng ngày |
| `action_*` | Tổng one-hot action của các event, tạo vector count 23 chiều |
| `session_count` | Số `session_id` khác nhau |
| `distinct_objects` | Số `object_key` khác nhau, bỏ qua event thiếu object |
| `event_count` | Tổng event trong ngày 0–34 |
| `active_days` | Số ngày có ít nhất một event |
| `first_active_day`, `last_active_day` | Ngày hoạt động đầu/cuối trong cửa sổ |
| `active_span_days` | `last_active_day - first_active_day + 1` |
| `days_since_last_activity` | `34 - last_active_day` |
| `active_day_ratio` | `active_days / 35` |
| `has_activity` | 1 nếu `event_count > 0`, ngược lại 0 |
| `object_missing_ratio` | Số event thiếu object chia `event_count` |
| `age` | `year(course_start) - birth`; ngoài `[10,100]` chuyển thành NA |
| `course_duration_days` | `date(end) - date(start)`; giá trị âm chuyển thành NA |

Có 4.539 `object` xuất hiện trong nhiều course, tối đa 5 course. Vì vậy
`object_key=(course_id, object)` là khóa đúng để tạo object hyperedge. Cache `v1`
hiện tại vẫn dùng `object` toàn cục và cần được tạo lại trước thí nghiệm chính.

## 7. Bảng feature quyết định sử dụng

### Input chính: `X_base` — 60 chiều

| Khối feature | Số chiều | NA/zero | Transform | Quyết định |
|---|---:|---|---|---|
| `day_00..day_34` | 35 | Không có event → 0 | `log1p`, rồi standardize bằng train | Dùng |
| `action_*` | 23 | Action không xảy ra → 0 | `log1p`, rồi standardize bằng train | Dùng |
| `session_count` | 1 | Không có event → 0 | `log1p`, rồi standardize bằng train | Dùng |
| `distinct_observed_objects` | 1 | Không có object → 0 | `log1p`, rồi standardize bằng train | Dùng |
| **Tổng** | **60** | | | |

`event_count` không đưa vào `X_base` vì bằng tổng `day_*` và cũng bằng tổng
`action_*`. Hai khối daily/action vẫn cùng được giữ: một khối mô tả thời gian, một
khối mô tả loại hành vi.

Kết quả Phase 3 có 3.305 node không phát sinh activity trong ngày 0–34; các count
của chúng bằng 0. `action_close_info` không xuất hiện trong dữ liệu có nhãn nên là
cột zero-variance, vẫn được giữ để cố định vocabulary 23 action. Mỗi seed có scaler
riêng, chỉ fit trên train của seed đó.

### Feature dùng cho ablation hoặc phân tích

| Feature | Xử lý NA và encoding | Quyết định |
|---|---|---|
| 8 activity summary: `event_count`, `active_days`, `active_span_days`, `first_active_day`, `last_active_day`, `days_since_last_activity`, `active_day_ratio`, `has_activity` | Node rỗng: first/last/span bằng 0, `days_since_last_activity=35`, `has_activity=0`; count dùng `log1p`; day chia 34; số ngày chia 35 | `X_augmented = X_base + 8`, chạy ablation |
| `object_missing_ratio` | Node rỗng → NA; train-median + cột missing | Audit/ablation, không vào input chính |
| `gender` | Null → `missing`; unseen → `other`; one-hot fit trên train | Context/fairness ablation |
| `education` | Null → `missing`; unseen → `other`; one-hot fit trên train | Context/fairness ablation |
| `age` | Ngoài `[10,100]` → NA; train-median + `age_missing`; standardize bằng train | Context/fairness ablation |
| `category` | Null → `missing`; unseen → `other`; one-hot fit trên train | Course-context ablation |
| `course_duration_days` | NA/âm → train-median + cột missing; standardize bằng train | Course-context ablation |
| `course_type` | Tất cả 247 course đều bằng 0 | Loại |
| `group_video`, `group_assignment`, `group_forum`, `group_web_page` | Là tổng các `action_*` tương ứng | Không dùng vì dư thừa |

Không one-hot `enroll_id`, `node_id`, `user_id`, `course_id`, `object` hoặc
`object_key`. Đây là khóa/index hoặc nguồn tạo hyperedge, không phải feature phân loại.

## 8. Hypergraph ban đầu

| Hyperedge | Cách tạo | Quy tắc chống leakage |
|---|---|---|
| Course | Các enrollment cùng `course_id` | Chỉ nối node trong tập đang xét |
| User | Các enrollment cùng `user_id` | Với split chính, mỗi user chỉ nằm trong một tập |
| Object | Các enrollment cùng `object_key`; bỏ singleton | Khóa phải gồm cả course; thống kê cardinality trên train |
| Behavioral | kNN trên `X_base` đã transform | Scaler và neighbor search chỉ fit trên train; chọn `k` bằng validation |

Audit theo khóa ghép hiện có:

| Object type | Số khóa | Singleton | Tỷ lệ singleton | Median enrollment/khóa | Max |
|---|---:|---:|---:|---:|---:|
| Video | 12.239 | 620 | 5,07% | 43 | 3.338 |
| Bài tập | 11.119 | 495 | 4,45% | 40 | 2.843 |
| Diễn đàn | 57.387 | 57.209 | 99,69% | 1 | 2 |
| Tổng | 80.745 | 58.324 | 72,23% | 1 | 3.338 |

Object diễn đàn gần như toàn singleton. Cấu hình chính vẫn giữ 178 khóa diễn đàn
không singleton toàn cục; khi tạo graph cho từng seed, luật `cardinality >= 2`
được áp dụng lại chỉ trên train. Hành vi diễn đàn vẫn được giữ trong `action_*`.

## 9. Kiểm tra bắt buộc trước khi train

```text
event_count = sum(day_00..day_34)
event_count = sum(action_*)
active_days <= 35
first_active_day <= last_active_day khi has_activity = 1
0 <= object_missing_ratio <= 1
user overlap giữa train/validation/test = 0
mọi scaler, median và vocabulary chỉ fit trên train
object_key chỉ thuộc một course và một object_type
```

Thứ tự xử lý:

```text
raw audit → canonical events → enrollment nodes → user-disjoint split
→ raw features → train-only transform → hyperedges → lưu cache → train
```
