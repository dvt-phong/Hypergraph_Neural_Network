# Plan: CFIN giữ hệt code tác giả, chỉ đọc X59 chia theo seed

Ngày: 2026-10-18. Em đã chốt (4 câu hỏi). Đã code; giữ `setdefault` cho CUDA_VISIBLE_DEVICES.

Gốc: github.com/wzfhaha/dropout_prediction, commit a99db2e. Patch: `baseline/CFIN/cfin_changes.patch`.

## 1. Quyết định

| Mục | Quyết định |
|---|---|
| Input | X59 của seed (`x59_s{seed}.npz`): train / val / test của seed; `feat_augment` gốc (mean / max theo người học và khóa học) |
| Nhánh context | bỏ (chỉ log activity): thay đổi kiến trúc duy nhất |
| Batch norm | **trả lại như gốc** (không sửa lỗi predict cập nhật thống kê BN) |
| Seed mô hình | **giữ `random_seed = 12345`** như gốc; không seed numpy / random (seed của mình chỉ đổi cách chia dữ liệu) |
| Train | train 60% của seed, 27 epoch, `early_stopping=False`, chấm epoch cuối; mỗi epoch in AUC của val (gốc in test) |

## 2. So với patch hiện tại

| Chỗ | Patch hiện tại | Sau khi sửa |
|---|---|---|
| `model.py` `batch_norm_layer` | hai nhánh BN dựng trong `tf.cond` (đã sửa lỗi) | **như gốc**: dựng ngoài `tf.cond` |
| `model.py` cuối `fit` | bỏ khối refit (chuỗi comment) và `saver.save("my_model/CFIN")` | **như gốc**: giữ khối comment và lưu `my_model/CFIN`, `return save_path` |
| `model.py` `get_feats`, `get_attn` | đã xoá | vẫn xoá: hai hàm đọc input context (u_*, c_*) đã bỏ, không được gọi |
| `main.py` `params["random_seed"]` | `arguments.seed` | **12345** như gốc |
| `main.py` `random.seed`, `np.random.seed` | có | **bỏ** (gốc không có) |
| `run_cfin.sh` | không tạo `my_model/` | tạo lại `my_model/` (gốc lưu model ở đó) |
| Còn lại | đọc X59 theo seed, bỏ context, 27 epoch, ghi kết quả / file dự đoán | giữ |

## 3. Một dòng môi trường cần em chốt

Gốc đặt cứng `os.environ['CUDA_VISIBLE_DEVICES'] = '2,4'`. Trên server chỉ có GPU 0 thì TensorFlow không thấy GPU và chạy CPU (không crash, chỉ chậm). Patch hiện tại đổi thành `setdefault` để `run_cfin.sh` chọn GPU 0.

Đề xuất: **giữ `setdefault`** (không đổi mô hình, chỉ chọn GPU).

## 4. Hệ quả cần biết

- Không seed numpy nên thứ tự batch mỗi lần chạy khác nhau: chạy lại cùng seed **không ra đúng cùng số** (như code gốc).
- Lỗi BN: mỗi lần `predict` (in val mỗi epoch, chấm val, chấm test) cập nhật thống kê BN bằng dữ liệu được chấm, nên kết quả phụ thuộc thứ tự chấm. Trong bài ghi: "CFIN is run with the authors' code unchanged, including its batch-normalization behaviour at inference".
- Kết quả CFIN có thể khác lần chạy vừa rồi (0.8750 ± 0.0015).

## 5. Kiểm tra

1. `git diff` của clone so với a99db2e: chỉ còn các thay đổi ở mục 2 (cột "Sau khi sửa").
2. Patch áp được lên clone sạch.
3. Smoke 2 epoch, 4,000 dòng: chạy hết, ghi `results.csv` và file dự đoán.

## 6. Chạy lại trên server

```bash
git pull
conda activate cfin
bash baseline/CFIN/run_cfin.sh setup
mv outputs/baselines/CFIN outputs/baselines/_CFIN_bn_fixed
nohup bash baseline/CFIN/run_cfin.sh full > outputs/logs/cfin.log 2>&1 &
```
