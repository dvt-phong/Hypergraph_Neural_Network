# Plan: chạy M và 6 baseline trên OULAD, cùng nguyên tắc với XuetangX

Ngày: 2026-10-19. Thay cho `docs/PLAN_OULAD_X60.md` (chỉ có M). Em duyệt Q1–Q3 theo đề xuất; đã code và kiểm tra ngày 2026-10-19.

## 1. Kiểm tra hiện trạng (chưa chạy được)

| # | Hiện trạng | Hệ quả |
|---|---|---|
| K1 | `baseline/NEW-DATASET/OULAD/` vẫn là bản cũ ngày 2026-10-09: X108 (có gender, age, region, …, module), split cũ (xáo seed 1, 80% → 80/20, 20% test), `TRAIN` cũ (dropout 0.5, wd 5e-4, eval 5, patience 40), kịch bản F1–F3 | khác hẳn XuetangX hiện tại (X59, 60/10/30 theo seed, lr 1e-3 / wd 1e-5 / dropout 0.3 / eval 1 / patience 200) |
| K2 | git theo dõi `baseline/new_dataset/OULAD/`, thư mục thật là `NEW-DATASET` (Windows phân biệt hoa thường với git) | `git status` báo 11 file bị xoá; commit lúc này sẽ xoá code OULAD khỏi git |
| K3 | `PLAN_OULAD_X60.md` chưa được code | `data/processed/oulad/` vẫn là train/validation/test.csv cũ, không có `events.csv`, `splits/`, `x*_s{seed}.npz` |
| K4 | Mọi baseline cố định tên file `x59_s{seed}.npz`; CFIN `FEATURE_COUNT = 59`; CA-TFHN `range(59)` và `org_context_feat_len: 59` | không đọc được dữ liệu OULAD |
| K5 | Thư mục ra của baseline cố định: `outputs/baselines/<model>`, CA-TFHN `code/datastore/s{seed}`, MST-GCN `code/cache/full` | chạy OULAD sẽ ghi đè kết quả và cache XuetangX |

Số đo trên `oulad.zip` (ngày 0–34): 32,593 enrollment, 28,785 người học, 22 khóa, tỉ lệ Withdrawn 0.3116; 2,525,357 dòng (site, ngày), 9,286,737 click; 18/20 loại activity có click; 4,530 enrollment không có click nào; 688,988 dòng trước ngày 0 (bỏ, như XuetangX bỏ d < 0).

## 2. Nguyên tắc (giống XuetangX)

| Mục | XuetangX (đang dùng) | OULAD |
|---|---|---|
| Nhãn | dropout của bộ dữ liệu | y = 1[final_result = Withdrawn] (Wu) |
| Node | mọi enrollment có nhãn | cả 32,593 enrollment |
| Cửa sổ | ngày 0–34 | ngày 0–34 (`date` của studentVle) |
| Feature | X59, chỉ log hoạt động, chung cho M và mọi baseline | X của mục 3, chung cho M và mọi baseline |
| Split | seed s: 30% test, rồi 1/7 phần còn lại làm val, phân tầng theo nhãn | giống hệt → 19,555 / 3,260 / 9,778 |
| Seed | 1, 11, 111, 1111, 11111 (chia + khởi tạo) | giống |
| Scale | log1p + z-score trên train của seed | giống |
| Hyperedge | Course, Object, User "any", self-loop | Course = module_presentation, Object = id_site, User = id_student "any", self-loop |
| M | lr 1e-3, wd 1e-5, dropout 0.3, hidden 128, 1000 epoch, eval 1, patience 200, chọn theo val AUC | giống |
| Baseline | siêu tham số gốc của tác giả; CFIN 27, CA-TFHN 15, MST-GCN 8 epoch, chấm epoch cuối; LR/SVM/XGBoost chọn theo val AUC | giống |
| Chấm điểm | AUC, precision, recall, F1 ở ngưỡng 0.5, mean ± std 5 seed | giống |

## 3. Feature: câu hỏi Q1

X59 của XuetangX = [35 số sự kiện theo ngày | tổng sự kiện | số object khác nhau | 22 loại action]. Bản dịch sang OULAD:

| Khối | X59 XuetangX | **X55 (đề xuất)** | X60 (plan 2026-10-17) |
|---|---|---|---|
| theo ngày | 35 | 35 click theo ngày | 35 |
| tổng | tổng sự kiện | tổng click | tổng click |
| object | số object khác nhau | số id_site khác nhau | không có |
| loại hành động | 22 action | 18 activity_type | 18 activity_type |
| bài kiểm tra | không có | không có | 6 cột assessment của Wu (n_assess, submitted_any, avg_score, weighted_score, mean/max lateness) |
| Tổng | 59 | **55** | 60 |

Đề xuất X55: đúng nguyên tắc "chỉ log hoạt động" như XuetangX (assessment là bảng điểm `studentAssessment.csv`, không phải log VLE), và giữ cột "số object" mà X60 đang thiếu. Cả hai cách đều công bằng giữa các mô hình vì mọi mô hình nhận cùng X. Nếu em chọn X60, mọi chỗ dưới đây đổi 55 → 60.

Mọi cột X55 là số đếm: x = (log(1 + c) − μ_train) / σ_train.

## 4. Dữ liệu (cùng định dạng `data/processed/simple/`)

```
data/processed/oulad/
  events.csv          node_id, enroll_id, user_id, course_id, label, action, object_id, course_day, clicks
                      1 dòng = (enrollment, id_site, ngày); action = activity_type; clicks = sum_click;
                      enrollment không có click = 1 dòng với action rỗng (như XuetangX)
  splits/s{seed}.csv  node_id, enroll_id, split
  x55_s{seed}.npz     enroll_id, user_id, course_id, label, split, X [32,593 × 55]
  hypergraph.npz      H (Course, Object, User), không phụ thuộc seed
```

Khác XuetangX duy nhất: thêm cột `clicks` (OULAD gộp sẵn click theo site và ngày). Số đếm X cộng `clicks` thay vì đếm dòng. MST-GCN và CA-TFHN đọc bằng `usecols`, nên cột thêm không ảnh hưởng; đồ thị của MST-GCN và H chỉ dùng việc có/không có cạnh.

File cũ trong `data/processed/oulad/` (train/validation/test, `*_assessment.csv`, `feature_names.csv`) xoá sau khi bản mới qua kiểm tra (hỏi em trước).

## 5. Code

### 5.1 M: `baseline/new_dataset/OULAD/` (không sửa `src/`)

Đổi tên `NEW-DATASET` → `new_dataset` (qua `git mv`, hết lỗi K2), rồi viết lại theo cấu trúc `src/` hiện tại:

| File | Việc |
|---|---|
| `0_config.py` | bỏ hằng số người học / khóa học / assessment và F1–F3; `TRAIN_RATIO / VALIDATION_RATIO / TEST_RATIO` 0.6 / 0.1 / 0.3; `TRAIN`, `SCENARIOS` (M, A1–A4, W1, L1, H, X1) như `src/0_config.py`; chỉ số cột X55 |
| `1_download.py` | giữ |
| `2_preprocess.py` | zip → `events.csv` + `splits/s{seed}.csv`; `split_enrollments` chép từ `src/2_preprocess.py` |
| `3_features.py` | như `src/3_features.py`: đếm một lần, mỗi seed scale trên train → `x55_s{seed}.npz` |
| `4_hypergraph.py` | đọc `events.csv` → `hypergraph.npz` |
| `5_graph_data.py` … `10_summary.py` | chép từ `src/`, chỉ đổi đường dẫn và tên file X |

Kết quả M: `outputs/oulad/results.csv`, `outputs/oulad/summary.csv`.

### 5.2 Baseline: câu hỏi Q2 (cách cho baseline đọc OULAD)

Đề xuất: thêm một biến `DATASET` (mặc định `xuetangx`, nên XuetangX không đổi gì):

| Chỗ sửa | Thay đổi |
|---|---|
| `ML/LR.py`, `SVM.py`, `XGBoost.py`, `summary.py` | thêm `--x-name` (mặc định `x59`); đã có `--x-dir`, `--output-dir` |
| 3 patch (CFIN, CA-TFHN, MST-GCN) | tên file X lấy từ tham số (mặc định `x59`); số cột lấy từ `X.shape[1]` thay cho số 59 cố định (CFIN `FEATURE_COUNT`, CA-TFHN `x_columns`, `org_context_feat_len`) |
| 3 `run_*.sh` | `DATASET=oulad` → `X_DIR=data/processed/oulad`, `X_NAME=x55`, `OUT=outputs/oulad/baselines/<model>`, CA-TFHN `datastore/oulad/s{seed}`, MST-GCN `cache/oulad` (hết lỗi K5) |

Không đổi mô hình hay siêu tham số của baseline nào. Patch phải tạo lại và thử trên bản clone sạch như lần trước.

### 5.3 Câu hỏi Q3: số epoch của baseline

OULAD có 19,555 enrollment train, ít hơn XuetangX khoảng 7 lần, nên với cùng số epoch gốc (8 / 15 / 27) baseline được cập nhật ít hơn khoảng 7 lần (MST-GCN batch 256: 77 bước × 8 = 616 bước). Đề xuất: giữ đúng số epoch gốc như XuetangX (nguyên tắc "siêu tham số của tác giả"), in val AUC theo epoch để xem có thiếu khớp không; chỉ đổi nếu em và thầy hướng dẫn muốn.

## 6. Kiểm tra (như XuetangX)

1. Mỗi seed: train / val / test rời nhau, hợp lại đủ 32,593, kích thước 19,555 / 3,260 / 9,778; tỉ lệ Withdrawn bằng nhau ở 3 tập; test khác nhau giữa các seed.
2. Tổng click trong `events.csv` = 9,286,737; 4,530 dòng rỗng; số đếm X55 trước scale trùng 54 cột tương ứng của X108 cũ (bitwise), cột số site tự kiểm bằng pandas.
3. X hữu hạn; mỗi cột có mean ≈ 0, std ≈ 1 trên train của seed.
4. Object ⊂ Course; H giống bản cũ.
5. Với cả 5 seed: input enrollment của mọi baseline = X55 (bitwise), phần thêm = 0; `summary.py` kiểm file test của baseline trùng tập test của npz.
6. XuetangX không đổi: chạy lại LR seed 1 và smoke 3 baseline với `DATASET` mặc định, ra đúng số cũ.
7. Smoke: M 10 epoch mọi kịch bản; LR/SVM/XGBoost seed 1; CFIN / CA-TFHN / MST-GCN smoke trên OULAD.

## 7. Lưu ý khi đọc kết quả

- 4,530 enrollment không có click (82.0% Withdrawn, đo 2026-10-19): mọi mô hình đều đoán dễ nhóm này, giống nhóm 3,305 enrollment rỗng của XuetangX.
- User "any" nối mọi enrollment của cùng người học, kể cả lần học sau: 1,309 enrollment có lần học lại cùng module ở kỳ sau, 91.2% Withdrawn (+ 8.4% Fail), so với 31.2% chung (đo 2026-10-19). Đây là cùng kiểu rò tương lai như User "any" trên XuetangX nhưng mạnh hơn; sau khi có M nên chạy A3 (bỏ User) để biết M phụ thuộc vào nó bao nhiêu.
- CA-TFHN: 22 khóa, mỗi khóa khoảng 1,500 enrollment, nên đồ thị bạn cùng khóa dày hơn XuetangX nhiều.

## 8. Lệnh chạy (sau khi code)

```bash
# M
cd baseline/new_dataset/OULAD
python 2_preprocess.py && python 3_features.py && python 4_hypergraph.py
python 9_train.py --scenario M --seeds 1 11 111 1111 11111 && python 10_summary.py
cd ../../..
# baseline
for m in LR SVM XGBoost; do python baseline/ML/$m.py --x-dir data/processed/oulad --x-name x55 --output-dir outputs/oulad/baselines/$m; done
DATASET=oulad bash baseline/CFIN/run_cfin.sh full
DATASET=oulad bash baseline/CA-TFHN/run_catfhn.sh features && DATASET=oulad bash baseline/CA-TFHN/run_catfhn.sh full
DATASET=oulad bash baseline/MST-GCN/run_mstgcn.sh features && DATASET=oulad bash baseline/MST-GCN/run_mstgcn.sh full
python baseline/summary.py --results-dir outputs/oulad/baselines --x-dir data/processed/oulad --x-name x55
```
