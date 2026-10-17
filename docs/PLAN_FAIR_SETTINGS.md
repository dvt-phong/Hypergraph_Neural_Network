# Plan: thống nhất seed và feature, siêu tham số theo tác giả

Ngày: 2026-10-17 (bản 2, thay bản "một bộ siêu tham số chung"). Em đã chọn phương án (b) cho số epoch; đã code.

## 1. Quyết định

| Mục | Quyết định |
|---|---|
| Thống nhất cho mọi mô hình | feature X59 (`x59_s{seed}.npz`), 5 seed 1, 11, 111, 1111, 11111 (seed chia train/val/test 60/10/30 phân tầng và khởi tạo, docs/PLAN_SEED_SPLIT.md), 4 chỉ số AUC / precision / recall / F1 ở ngưỡng 0.5, báo cáo mean ± std |
| Siêu tham số của baseline | giữ đúng giá trị của tác giả, kể cả số epoch cố định; chấm epoch cuối |
| Siêu tham số của M | của em, chọn theo val AUC (mục 3) |

## 2. Baseline: siêu tham số của tác giả

Đã đối chiếu: code đã dùng siêu tham số gốc của tác giả; chỉ sửa số epoch (dưới).

| Baseline | Siêu tham số gốc (giữ nguyên) | Chỉ khác bản gốc ở |
|---|---|---|
| MST-GCN | lr 1e-4, wd 1e-5, dropout 0.3, batch 256, hidden 128, 2 lớp | input X59; số epoch (dưới) |
| CA-TFHN | lr 1e-4, batch 256, GraphSage 32/16, LSTM 128/32, attention 64/16, DNN 16-8-4 | input X59 nên `lstm_input_features` 184 → 16 và context 7 → 59 cột (do kích thước dữ liệu); số epoch (dưới) |
| CFIN | lr 1e-4, L2 1e-5, dropout 0.1 (keep 0.9), batch 32, embedding 32, conv 512, deep 256 | bỏ nhánh context user/course (chỉ activity, theo PLAN_X59); số epoch (dưới) |
| LR, SVM | không có tác giả gốc: chọn alpha theo val AUC, tối đa 200 epoch, patience 10 | — |
| XGBoost (file mới, chưa commit) | chọn max_depth theo val AUC, tối đa 3000 cây, patience 50 | — |

**Số epoch (phương án b, em chọn).** Đúng như tác giả: MST-GCN 8, CA-TFHN 15, CFIN 27 epoch cố định, không early stopping, chấm val và test một lần bằng trọng số của epoch cuối. Val AUC vẫn được in sau mỗi epoch để theo dõi (bản gốc của CA-TFHN và CFIN in test), không dùng để chọn. Trước đó code chạy tối đa 24 / 45 / 81 epoch và chọn checkpoint theo val AUC.

## 3. M: siêu tham số của em

| Siêu tham số | Hiện tại trong `TRAIN` | Đổi thành |
|---|---|---|
| learning_rate | 1e-3 | 1e-3 |
| weight_decay | 5e-4 | **1e-5** |
| dropout | 0.5 | **0.3** |
| hidden_dim | 128 | 128 |
| epochs | 1000 | 1000 |
| eval_every | 5 | **1** |
| patience | 40 (lần đánh giá = 200 epoch) | **200** (= 200 epoch) |
| family_weight_lr | 0.05 | 0.05 |
| batch | full-batch | full-batch |

Bằng chứng (seed 1, chọn theo val AUC):

| Cấu hình | best_epoch / epochs_run | val AUC | test AUC |
|---|---|---|---|
| hiện tại (wd 5e-4, dropout 0.5) | 995 / 1000 (chạm trần) | 0.8766 | 0.8764 |
| lr 1e-4, wd 1e-5, dropout 0.3 | 1500 / 1500 (chạm trần) | 0.8748 | 0.8742 |
| **lr 1e-3, wd 1e-5, dropout 0.3** | **410 / 610 (tự dừng)** | **0.8769** | **0.8767** |

Lý do đổi: cấu hình mới tự hội tụ (dừng trước trần), val AUC ngang hoặc hơn chút (chênh 0.0003 ≈ std giữa các seed).

## 4. Code

| File | Việc |
|---|---|
| `src/0_config.py` | sửa 4 giá trị của `TRAIN`: weight_decay 1e-5, dropout 0.3, eval_every 1, patience 200 |
| MST-GCN | `trainer.py`: bỏ chọn checkpoint và `--patience`; `run_mstgcn.sh`: bỏ `--num_epochs 24 --patience 5` (mặc định 8) |
| CA-TFHN | `train.py`: `-e` mặc định 15, bỏ `-patience` và chọn checkpoint; giữ khởi tạo lại mô hình chết |
| CFIN | `main.py`: `model.fit(..., early_stopping=False)` với 27 epoch (`params["epoch"]`), bỏ checkpoint `my_model/` |
| `README.md` | cập nhật siêu tham số của M và ghi chú baseline |
| `docs/PLAN_FAIR_SETTINGS.md` | file này |

Các kịch bản ablation (A1–A4, W1, L1, H) dùng chung `TRAIN` mới.

## 5. Chạy và kiểm tra

1. Server: `bash scripts/run_tmux.sh --scenario M --seeds 1 11 111 1111 11111` (không cần `sed` nữa; khoảng 30 phút mỗi seed).
2. Seed 1 phải ra val AUC ≈ 0.8769, best_epoch ≈ 410 (giống lần chạy B).
3. Baseline sâu (MST-GCN, CA-TFHN, CFIN) chạy bằng `run_*.sh` như hiện tại.
4. Với baseline, `best_epoch` = `epochs_run` = 8 / 15 / 27 (epoch được chấm).
