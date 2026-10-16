# Plan: X59 (chỉ log activity) + 5 seed, chung cho mọi mô hình

Ngày: 2026-10-16. Đã chốt với em.

## 1. Quyết định

| Mục | Quyết định |
|---|---|
| Feature | X59 = 35 số event theo ngày + total_events + distinct_objects + 22 action; x = (log(1 + c) − μ_train) / σ_train. Bỏ gender, education, age, category |
| Test | official test cố định (67,699 enrollment) |
| Train / val | seed s chia official train (157,943) 80 : 20 theo enrollment, ngẫu nhiên bằng seed s |
| Seed | 1, 11, 111, 1111, 11111; seed s vừa chia train/val vừa khởi tạo mô hình; báo cáo mean ± std |
| Input chung | `data/processed/simple/x59_s{seed}.npz`: mọi mô hình (M và baseline) đọc cùng file |
| Baseline | LR, SVM, CFIN (chỉ activity), MST-GCN, CA-TFHN (chỉ X59); bỏ SIG-Net |
| Kịch bản M | M, A1–A4, W1, L1, H (+ X1 tùy chọn); bỏ F1–F3 |

Cấu trúc đồ thị (Course / Object / User hyperedge của M, đồ thị của MST-GCN, CA-TFHN) giữ nguyên vì không phải feature.

## 2. Dữ liệu

```
data/processed/simple/
  events.csv            node_id, enroll_id, user_id, course_id, label, action, object_id, course_day
                        (mọi 225,642 enrollment, event ngày 0–34; enrollment không có event = 1 dòng trống)
  splits/s{seed}.csv    node_id, enroll_id, split   (train / validation / test)
  x59_s{seed}.npz       enroll_id, user_id, course_id, label, split, X (225,642 × 59), theo thứ tự node_id
  hypergraph.npz        H (Course, Object, User), không phụ thuộc seed
```

node_id = vị trí của enroll_id trong danh sách enroll_id tăng dần.

## 3. Code

| File | Việc |
|---|---|
| `src/2_preprocess.py` | raw → `events.csv` + 5 file split |
| `src/3_features.py` | `events.csv` → số đếm 59 cột → với mỗi seed: log1p + z-score trên train → `x59_s{seed}.npz` |
| `src/4_hypergraph.py` | `events.csv` → `hypergraph.npz` |
| `src/5_graph_data.py` | `hypergraph.npz` + `x59_s{seed}.npz` của một seed |
| `src/9_train.py` | với mỗi seed: đọc dữ liệu seed đó, chạy các kịch bản |
| `baseline/ML/LR.py`, `SVM.py` | đọc `x59_s{seed}.npz` |
| `baseline/CFIN` | đọc `x59_s{seed}.npz`; bỏ nhánh context user/course trong model; giữ phần trung bình / max theo user và course của chính activity (feat_augment gốc) |
| `baseline/MST-GCN` | đồ thị đọc từ `events.csv`; feature enrollment = X59 |
| `baseline/CA-TFHN` | chuỗi 5 tuần × 7 ngày từ 35 cột ngày của X59; org_context = X59; link prediction dùng trung bình X59 theo user / course thay cho demographic |
| `baseline/summary.py` | bảng mean ± std của 5 baseline |

Xóa: `baseline/prepare_raw.py`, `baseline/SIG-Net/`, mọi `docs/*.md` cũ, phần feature user/course trong src.

## 4. Kiểm tra

- [ ] split s1 trùng split cũ (seed 1); mỗi split: train ∩ val ∩ test = ∅, test = 67,699
- [ ] số event trong `events.csv` = tổng số event của 3 CSV cũ
- [ ] X59 của seed 1 = 59 cột đầu của X89 cũ (cùng train)
- [ ] `hypergraph.npz` mới = H cũ sau khi đổi node id
- [ ] smoke: M 10 epoch mọi kịch bản; LR, SVM, CFIN local; MST-GCN, CA-TFHN trên server

## 5. Lệnh chạy

```
python src/2_preprocess.py
python src/3_features.py
python src/4_hypergraph.py
python src/9_train.py --scenario M --seeds 1 11 111 1111 11111      # chỉ mô hình M; ablation chạy sau
python src/10_summary.py

python baseline/ML/LR.py
python baseline/ML/SVM.py
bash baseline/CFIN/run_cfin.sh setup|smoke|full
bash baseline/MST-GCN/run_mstgcn.sh setup|features|smoke|full
bash baseline/CA-TFHN/run_catfhn.sh setup|features|smoke|full
python baseline/summary.py
```

## 6. Ghi chú (2026-10-16, sau khi chạy thử local)

**Seed s (1, 11, 111, 1111, 11111) làm 2 việc, giống nhau cho mọi mô hình:**
1. Chia: xáo official train (157,943, sắp theo enroll_id) bằng `random.Random(s)`, 80% đầu → train (126,354), 20% sau → validation (31,589). Official test (67,699) giữ nguyên. Ghi một lần vào `splits/s{s}.csv`, rồi vào `x59_s{s}.npz`.
2. Train: mô hình khởi tạo và xáo batch bằng seed s.

Mỗi mô hình chạy 5 lần (một lần mỗi seed), báo cáo mean ± std của 5 lần trên test.

**Input giống nhau (đã kiểm tra bitwise, seed 1–11111):** M, LR, SVM, CFIN (59 field gốc), MST-GCN (feature
enrollment), CA-TFHN (org_context, seq_feat = 35 cột ngày) đọc đúng X59 và đúng train / val / test của
`x59_s{seed}.npz`. Phần mỗi phương pháp tự thêm (đều tính từ cùng log activity, không có demographic):
M: hypergraph; CFIN: mean / max X59 theo user và course; MST-GCN: đồ thị enrollment–object–course,
feature của node object / course tính từ log; CA-TFHN: đồ thị classmates từ link prediction trên mean X59.

**CA-TFHN AUC 0.5 – nguyên nhân và cách sửa:** classifier 16 → 16 → 8 → 4 → 1 (ReLU); lúc khởi tạo đầu
vào của nó gần như giống nhau giữa các enrollment, nên với một số seed mọi unit của lớp 4 bằng 0 cho
mọi enrollment (seed 1: 0/4; 2/20 seed thử) → gradient 0, chỉ bias học, AUC 0.5. Sửa trong train.py:
kiểm tra ngay lúc khởi tạo (mỗi lớp ReLU phải có ≥ 1 unit > 0 trên batch validation đầu), nếu không thì
khởi tạo lại với seed r + 1000·k; vẫn giữ kiểm tra sau epoch 1 (p hằng số, std < 1e-6).
Chạy local seed 1 (torch 1.11, PyG 2.1 như server): test AUC 0.8709 (bài gốc 0.8707).

**MST-GCN:** đồ thị DGL nhận `num_nodes_dict`, nếu không enrollment không có object (không có event) bị mất node.

**Server:** `git pull`, rồi `bash baseline/<model>/run_<model>.sh setup` cho từng baseline (setup luôn đưa
clone về commit gốc và áp patch mới), rồi `features` (MST-GCN, CA-TFHN), `smoke`, `full`.
M: `bash scripts/run_tmux.sh --scenario M --seeds 1 11 111 1111 11111`.
