# Plan: thêm dataset OULAD (tải về + xử lý + chạy cùng pipeline)

Ngày 2026-10-08. Chỉ là plan, chưa sửa code. Các số dưới đây **đã đo** trên bản OULAD tải
về scratchpad (không đụng `data/`).

## 0. OULAD là gì (tóm tắt đã kiểm tra)

- Nguồn: Kuzilek, Hlosta, Zdrahal (2017), *Open University Learning Analytics dataset*,
  Scientific Data 4:170171. License CC-BY 4.0.
- Link tải dùng được: `https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip`
  (HTTP 200, 46,7 MB, 7 CSV + OULAD.names). Link cũ của OU (`analyse.kmi.open.ac.uk/open_dataset/download`)
  giờ chuyển hướng sang trang HTML, không tải được file.
- 7 bảng: `courses` (22 module-presentation), `studentInfo` (32.593 enrollment, 28.785 sinh viên),
  `studentRegistration`, `vle` (6.364 tài liệu `id_site`, 20 `activity_type`), `studentVle`
  (10.655.280 dòng, mỗi dòng = sinh viên × tài liệu × ngày × `sum_click`), `assessments`, `studentAssessment`.
- `date` tính theo ngày từ lúc module bắt đầu (−25 … 269). `final_result`: Pass 12.361, Withdrawn 10.156,
  Fail 7.052, Distinction 3.024. Giá trị thiếu ghi bằng `?`.
- Không có test chính thức (khác XuetangX) → mình tự chia.

## 1. Ánh xạ OULAD vào mô hình hiện tại

```
XuetangX                         OULAD
enrollment (node)          <->   (id_student, code_module, code_presentation)
course_id   -> Course      <->   code_module + code_presentation, vd "BBB_2013J"   (22 hyperedge)
object      -> Object      <->   id_site (mỗi id_site thuộc đúng 1 presentation -> Object ⊂ Course, đã kiểm tra)
username    -> User        <->   id_student
event       -> 1 dòng log  <->   1 dòng studentVle, đếm bằng sum_click
action (22)                <->   activity_type (20)
```

Bước 5–8 (graph, HGNN, MLP, model) **không đổi**. Chỉ bước 1–4, 9, 10 cần biết dataset.

## 2. Các quyết định cần em chọn (thầy đề xuất cái đầu tiên)

| # | Câu hỏi | Đề xuất | Lựa chọn khác | Số đo |
|---|---|---|---|---|
| D1 | Nhãn dropout | **Withdrawn = 1, còn lại (Pass, Distinction, Fail) = 0** — đúng nghĩa "bỏ học" | Withdrawn + Fail = 1 ("at-risk") | Tỉ lệ dương sau D2: 17,7% (D1 đề xuất) / 43,6% (lựa chọn khác) |
| D2 | Bỏ sinh viên đã rút trước khi hết cửa sổ quan sát | **Bỏ** Withdrawn có `date_unregistration < 35` | Giữ lại | 5.335 node bị bỏ (2.676 rút trước ngày 0). Nếu giữ thì nhãn đã lộ ngay trong cửa sổ → AUC ảo |
| D3 | Cửa sổ quan sát | **Ngày 0–34 (35 ngày)**, giống hệt XuetangX | Thêm 1 cột click trước ngày 0 | 23,7% số dòng studentVle nằm trong [0, 35); 6,5% trước ngày 0 |
| D4 | Chia dữ liệu | **Ngẫu nhiên theo enrollment, seed 1: test 20%, phần còn lại chia train/val 80/20** (cùng luật 80/20 với XuetangX) | Theo thời gian: train 2013, test 2014 | n = 27.258 → train 17.444 / val 4.362 / test 5.452 |
| D5 | Luật User hyperedge | **Cùng sinh viên + cùng presentation** (học song song cùng kỳ) | "any" như XuetangX | Xem mục 3 — "any" lộ tương lai rất rõ trên OULAD |
| D6 | Điểm bài kiểm tra (assessments) | **Chưa dùng** ở bản đầu, để X giống cấu trúc XuetangX (chỉ hành vi + bối cảnh) | Thêm khối điểm TMA/CMA nộp trước ngày 35 | Cả 22/22 presentation có ít nhất 1 bài đến hạn trước ngày 35 |

## 3. Cảnh báo quan trọng: User "any" lộ nhãn trên OULAD

Trên OULAD, sinh viên **chỉ học lại một module khi lần trước không qua**. Đo trên 27.258 node (sau D2):

- 816 enrollment có một lần học lại cùng module ở kỳ sau. Kết quả của lần trước: Withdrawn 722, Fail 94,
  **Pass/Distinction 0**.
- Tức là 88,5% các node này là dropout, trong khi tỉ lệ chung chỉ 17,7%. Với User "any", node test
  ở 2013J nối với node 2014J của chính sinh viên đó (có `num_of_prev_attempts = 1`) → mô hình "nhìn thấy"
  việc học lại trong tương lai.
- Luật "cùng presentation" chỉ nối các enrollment học song song cùng kỳ: 616 hyperedge, 1.232 node
  (4,5% node). Ít, nên A3 (bỏ User) trên OULAD gần như chắc sẽ ≈ M — đó là kết quả trung thực.
- Gợi ý: báo với thầy hướng dẫn, vì cùng lý lẽ này áp dụng cho XuetangX (đang dùng "any").

## 4. Feature X của OULAD (112 cột)

Cùng cách làm như XuetangX: hành vi log1p + z-score (thống kê chỉ từ train), one-hot luôn có thêm 1 cột
"missing", không có cột "other" (giá trị lạ → dừng chương trình).

| Khối | Cột | Số cột |
|---|---|---|
| Hành vi `[0, 57)` | 35 cột tổng click theo ngày 0–34 · total_clicks · distinct_sites · 20 cột click theo activity_type | 57 |
| Người học `[57, 101)` | one-hot: gender 2+1, region 13+1, highest_education 5+1, imd_band 10+1 (`?` = 1.111 node), age_band 3+1, disability 2+1 · số (z-score): num_of_prev_attempts, studied_credits, date_registration (45 thiếu → giá trị trung bình train, tức z = 0) | 44 |
| Khóa học `[101, 112)` | one-hot code_module 7+1, kỳ B/J 2+1 | 11 |

**Không dùng** (vì là nhãn hoặc lộ nhãn): `final_result`, `date_unregistration`; `studentAssessment` và
`assessments` chưa dùng (D6). Node không có click nào trong 0–34 vẫn giữ (hàng hành vi = 0), như XuetangX:
1.050 node.

F1/F2/F3 giữ nguyên nghĩa: F1 = hành vi (57), F2 = hành vi + người học (101), F3 = hành vi + khóa học (68).

## 5. Sửa code theo từng file

Nguyên tắc: thêm tham số `--dataset {xuetangx,oulad}`, **mặc định `xuetangx`** → mọi lệnh XuetangX cũ
chạy y như bây giờ, đường dẫn XuetangX không đổi (`data/processed/simple`, `outputs/results.csv`).

| File | Thay đổi |
|---|---|
| `0_config.py` | Thêm `OULAD_RAW = data/raw/oulad`, `OULAD_PROCESSED = data/processed/oulad`, `OULAD_OUTPUTS = outputs/oulad`; URL UCI; 20 activity_type; các từ vựng one-hot (gender, region, education, imd_band, age_band, disability, module, semester); `FEATURE_BLOCKS = {"xuetangx": (59, 71, 89), "oulad": (57, 101, 112)}` (điểm cắt hành vi / người học / tổng); luật User theo dataset |
| `1_download.py` | Viết `download_oulad`: tải zip UCI về `data/raw/oulad/oulad.zip` (bỏ qua nếu đã có), kiểm tra đủ 7 CSV; thêm `--dataset` |
| `2_preprocess.py` | Thêm `preprocess_oulad`: đọc CSV thẳng từ zip (không giải nén, giống cách đọc tar.gz), ghép studentInfo + studentRegistration + vle, `enroll_id` = số thứ tự dòng trong studentInfo, áp D1/D2, chia D4, ghi `train/validation/test.csv` gồm các cột chung (`node_id, enroll_id, user_id, course_id, label, action, object_id, course_day`) + `clicks` + các cột bối cảnh OULAD |
| `3_features.py` | Thêm `build_split_features_oulad` (đếm theo `clicks`); phần log1p + z-score dùng chung; `feature_columns(name, dataset)` lấy điểm cắt từ `FEATURE_BLOCKS` |
| `4_hypergraph.py` | `read_split`: với OULAD mọi activity_type đều là object (key `course|activity_type|id_site`); `user_hyperedges`: key `user|id` (XuetangX "any") hoặc `user|id|presentation` (OULAD, D5) |
| `5_graph_data.py` | `apply_scenario` truyền `dataset` vào `feature_columns` |
| `9_train.py` | `--dataset` chọn thư mục dữ liệu và file kết quả `outputs/oulad/results.csv` |
| `10_summary.py` | `--dataset` chọn `results.csv` / `summary.csv` / `ket_qua.xlsx` tương ứng |
| `6_hgnn`, `7_mlp`, `8_model` | Không đổi (số chiều lấy từ X) |

Siêu tham số giữ nguyên `TRAIN` của XuetangX, không tune riêng cho OULAD.

Lưu ý: em đang có thay đổi chưa commit ở `0_config.py` (bỏ bớt comment) và `1_download.py`
(stub `download_oulad`). Thầy sẽ code chồng lên đúng bản đó.

## 6. Kiểm tra sau khi code

- O1. Sau bước 2: 27.258 node; train/val/test = 17.444 / 4.362 / 5.452; không trùng enrollment giữa 3 file;
  tỉ lệ dropout từng split ≈ 17,7%.
- O2. Tổng `clicks` trong 3 CSV = tổng `sum_click` của studentVle với 0 ≤ date < 35 trên các node giữ lại.
- O3. X: 112 cột, hữu hạn; cột hành vi của train có mean 0 / std 1 (cột hằng số → 0).
- O4. Mỗi Object hyperedge nằm trong đúng 1 Course hyperedge; mỗi User hyperedge chỉ trong 1 presentation.
- O5. Phép thử C7: đảo toàn bộ nhãn val/test → loss và trọng số khi train giống hệt (1 thread).
- O6. XuetangX không đổi: chạy `9_train.py --scenario M --epochs 10 --seeds 1` (mặc định xuetangx) trước
  và sau khi sửa → cùng số.
- O7. Smoke: `9_train.py --dataset oulad --scenario all --epochs 10 --seeds 1` chạy hết 11 kịch bản.

## 7. Lệnh chạy trên server (sau khi duyệt và code xong)

```
python src/1_download.py   --dataset oulad
python src/2_preprocess.py --dataset oulad
python src/3_features.py   --dataset oulad
python src/4_hypergraph.py --dataset oulad
python src/9_train.py      --dataset oulad --scenario M                  # 5 seed trước
python src/9_train.py      --dataset oulad --scenario A1 A2 A3 A4 W1 F1 F2 F3 L1 H
python src/10_summary.py   --dataset oulad
```

## 8. Ngoài phạm vi plan này

Baseline trên OULAD (trong `baseline/` không có code nào xử lý OULAD — đã grep), chia theo thời gian,
feature điểm số. Làm sau nếu em cần.
