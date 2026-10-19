# Plan: CA-TFHN trên OULAD có precision / recall / F1 = 0 ở 3/5 seed

Ngày: 2026-10-19. Cách sửa đã chọn: docs/PLAN_CATFHN_OULAD_v2.md (cùng số bước cập nhật, chỉ CA-TFHN).

## 1. Hiện tượng (server, 5 seed, 15 epoch gốc, ngưỡng 0.5)

| Seed | 1 | 11 | 111 | 1111 | 11111 |
|---|---|---|---|---|---|
| test AUC | 0.7958 | 0.7894 | 0.7898 | 0.7886 | 0.8015 |
| test F1 | 0 | 0.5621 | 0 | 0 | 0.6099 |

AUC bình thường, nhưng ở 3 seed mọi xác suất đều < 0.5, nên mô hình đoán "không bỏ học" cho tất cả.

## 2. Chẩn đoán (máy em, seed 1, dữ liệu OULAD thật, bản sao `train.py` trong scratchpad)

### 2.1 Theo epoch (40 epoch, xác suất trên validation)

| Epoch | val AUC | p trung vị | p lớn nhất | tỉ lệ p ≥ 0.5 | F1 @0.5 |
|---|---|---|---|---|---|
| 1 | 0.6798 | 0.380 | 0.384 | 0 | 0 |
| 5 | 0.7536 | 0.215 | 0.444 | 0 | 0 |
| 15 (gốc) | 0.7819 | 0.212 | 0.470 | 0 | 0 |
| 32 | 0.7934 | 0.204 | 0.4995 | 0 | 0 |
| 33 | 0.7963 | 0.212 | 0.5012 | 0.189 | 0.591 |
| 40 | 0.7946 | 0.206 | 0.513 | 0.204 | 0.604 |

Từ epoch 7, p90 = p99 = p lớn nhất: khoảng 19% node có **cùng một xác suất** (cái "trần"), và trần chỉ tăng khoảng 0.0017 mỗi epoch.

### 2.2 Cái trần là gì

Ở epoch 15: 507/3,260 node validation (80.3% Withdrawn, gồm 92% số enrollment không có click) có cùng p = 0.470.

Bộ phân loại của tác giả: 16 → ReLU → 16 → ReLU → 8 → ReLU → 4 → 1.

| Lớp ReLU | Node ở trần | Các node khác |
|---|---|---|
| ReLU cuối (4 unit) | **0 unit sống ở cả 507 node** | ≥ 1 unit sống ở mọi node |
| Đầu vào bộ phân loại | 507 hàng khác nhau | — |

Với nhóm rủi ro cao, lớp ReLU cuối tắt hết, nên logit = bias đầu ra, p = σ(b), bất kể đầu vào. Thông tin vẫn có ở đầu vào nhưng bị xoá ở lớp này. Bias học với Adam lr 1e-4, khoảng 77 bước mỗi epoch, nên trần nhích chậm. Seed nào có bias khởi tạo cao (11, 11111) thì trần qua 0.5 trong 15 epoch, seed khác thì không.

Đây cùng là hiện tượng ReLU chết đã gặp trên XuetangX (AUC 0.5), nhưng ở đây chỉ chết với một nhóm node.

### 2.3 Hai nguyên nhân tách riêng

| Thử | test AUC | p lớn nhất | node cùng p lớn nhất | F1 @0.5 |
|---|---|---|---|---|
| Gốc, 15 epoch | 0.7977 | 0.470 | 507 | 0 |
| ReLU cuối → LeakyReLU(0.01), 15 epoch | 0.7996 | 0.472 | 1 | 0 |
| Gốc, 40 epoch | 0.8017 | 0.513 | — | 0.6148 |
| Gốc, 15 epoch, ngưỡng chọn trên val (0.265) | 0.7977 | — | — | 0.6276 |

1. **Chính:** sau 15 epoch, mô hình còn "rụt rè": xác suất bị nén dưới 0.5. OULAD nhỏ hơn XuetangX khoảng 7 lần (77 so với 529 bước mỗi epoch), và Withdrawn là lớp thiểu số (31%). Trên XuetangX, dropout là lớp đa số (76%) nên p tự nhiên > 0.5 và lỗi này bị che. LeakyReLU bỏ được trần nhưng **không** làm F1 hết bằng 0.
2. **Phụ:** trần do ReLU cuối chết làm 16–19% node bị hoà nhau, nên mất một chút AUC (khoảng 0.002 ở seed 1).

Code không có lỗi: dữ liệu, nhãn và mask đều đúng (đã kiểm ở 5 seed). Đây là hành vi của mô hình gốc với số epoch gốc trên một tập nhỏ có lớp dương là thiểu số.

## 3. Cách sửa (em chọn)

| | Cách | Đổi gì | Ưu | Nhược |
|---|---|---|---|---|
| **S1 (đề xuất)** | Ngưỡng chọn trên validation cho **mọi** mô hình (M và 6 baseline): t = argmax F1 trên val của seed, áp nguyên cho test | `9_train.py` (src + OULAD) lưu `seed{s}_{validation,test}.csv` như baseline; `baseline/summary.py --threshold val` đã có; thêm M vào bảng | không đổi mô hình hay epoch nào, AUC giữ nguyên, công bằng, chạy lại M rất nhanh | phải báo cáo 2 bảng (ngưỡng 0.5 và ngưỡng val) |
| S2 | CA-TFHN train cùng số **bước cập nhật** như trên XuetangX: 15 × 529 / 77 ≈ 103 epoch, chấm epoch cuối | `run_catfhn.sh`: `-e 103` khi `DATASET=oulad` | sát ý "15 epoch của tác giả" theo số bước | lệch số epoch gốc; phải hỏi lại CFIN (27) và MST-GCN (8) có làm tương tự không |
| S3 | Đổi ReLU cuối thành LeakyReLU | `src/models.py` | bỏ trần | đổi kiến trúc của tác giả, F1 vẫn = 0 → không đề xuất |

Đề xuất S1. Trong bài, bảng chính vẫn để ngưỡng 0.5; bảng ngưỡng chọn trên validation đi kèm và ghi chú CA-TFHN. Thầy cũng ghi rõ cái trần trong phần mô tả baseline.

## 4. Kiểm tra sau khi sửa (S1)

1. M seed 1 OULAD và XuetangX: AUC, F1 @0.5 giữ đúng số cũ (cùng seed, cùng code huấn luyện; chỉ thêm phần ghi file).
2. `summary.py`: file test của M khớp tập test của npz; ngưỡng val của mỗi seed nằm trong (0, 1).
3. CA-TFHN: không chạy lại, dùng file dự đoán đã lưu trên server.
