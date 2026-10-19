# Plan v2 (nhanh gọn): CA-TFHN trên OULAD

Ngày: 2026-10-19. Thay mục 3 của `docs/PLAN_CATFHN_OULAD.md` (nguyên nhân giữ nguyên). Em chọn: chỉ áp dụng cho CA-TFHN (CFIN, MST-GCN giữ epoch gốc); đã code ngày 2026-10-19.

## Ý tưởng

CA-TFHN trên OULAD được **cùng số bước cập nhật** như trên XuetangX, thay cho cùng số epoch:

| | train | batch | bước / epoch | epoch | bước cập nhật |
|---|---|---|---|---|---|
| XuetangX | 135,384 | 256 | 529 | 15 (gốc) | 7,935 |
| OULAD hiện tại | 19,555 | 256 | 77 | 15 | 1,155 |
| **OULAD mới** | 19,555 | 256 | 77 | **103** | 7,931 |

103 = round(15 × 529 / 77). Mô hình, lr, batch, ngưỡng 0.5 và cách chấm epoch cuối đều giữ nguyên. Trong bài sẽ ghi: "trained for the same number of updates as the authors' 15 epochs on XuetangX".

Bằng chứng (seed 1, máy em): F1 @0.5 bằng 0 cho đến epoch 32; từ epoch 33 thì xác suất cao nhất vượt 0.5 và F1 khoảng 0.6. Ở epoch 40: test AUC 0.8017, F1 0.6148.

## Sửa code (1 file)

`baseline/CA-TFHN/run_catfhn.sh`: thêm `EPOCHS` (XuetangX 15, OULAD 103), truyền `-e "$EPOCHS"` cho bước `full`. Không đổi patch, không đổi `train.py`, nên XuetangX không đổi.

## Chạy lại trên server (không cần dựng lại features)

```bash
git pull
mv outputs/oulad/baselines/CA-TFHN outputs/oulad/baselines/_CA-TFHN_15epoch   # giữ bản cũ để đối chiếu
conda activate catfhn
DATASET=oulad bash baseline/CA-TFHN/run_catfhn.sh full
python baseline/summary.py --results-dir outputs/oulad/baselines --x-dir data/processed/oulad --x-name x55
```

Ước lượng: 15 epoch mất 1.55 phút, nên 103 epoch khoảng 11 phút mỗi seed, khoảng 55 phút cho 5 seed.

## Kiểm tra sau khi chạy

1. Cả 5 seed có F1 > 0 (có node có p ≥ 0.5).
2. AUC không giảm so với bản 15 epoch (0.7930 ± 0.0055).
3. Nếu vẫn còn seed có F1 = 0: báo lại thầy. Khi đó dùng thêm bảng ngưỡng chọn trên validation (S1 của plan cũ).

## Câu hỏi duy nhất

Có áp quy tắc "cùng số bước cập nhật" cho CFIN (batch 32: 27 epoch → 187) và MST-GCN (batch 256: 8 epoch → 55) trên OULAD không?
- Không (nhanh nhất): chỉ CA-TFHN. Lý do ghi trong bài: CFIN và MST-GCN không bị F1 = 0, nên giữ epoch gốc.
- Có (nhất quán hơn): thêm một dòng tương tự vào `run_cfin.sh` và `run_mstgcn.sh`, chạy lại khoảng 38 phút (CFIN) / khoảng 39 phút (MST-GCN) mỗi seed.
