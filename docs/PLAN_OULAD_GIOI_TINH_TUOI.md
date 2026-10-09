# Thêm giới tính và tuổi vào X của OULAD

Ngày 2026-10-09. Chỉ sửa code trong `baseline/new_dataset/OULAD/` (src không đổi).

## 1. Dữ liệu hiện có

`gender` và `age_band` đã nằm sẵn trong `train.csv`, `validation.csv`, `test.csv` (bước 2 đã ghi), nên **không cần chạy lại 2_preprocess**.

Đếm theo node (đo trên data/processed/oulad):

| Split | Nodes | gender F / M | age_band 0-35 / 35-55 / 55<= | Thiếu |
|---|---|---|---|---|
| train | 20,859 | 9,388 / 11,471 | 14,777 / 5,943 / 139 | 0 |
| validation | 5,215 | 2,343 / 2,872 | 3,651 / 1,533 / 31 | 0 |
| test | 6,519 | 2,987 / 3,532 | 4,516 / 1,957 / 46 | 0 |

Tỉ lệ Withdrawn trên train: F 0.308, M 0.312; 0-35: 0.321, 35-55: 0.285, 55<=: 0.230.

## 2. Cách xử lý (theo XuetangX)

| Field | XuetangX | OULAD (đề xuất) |
|---|---|---|
| gender | one-hot female, male + cột missing (vì 57.5% thiếu) | one-hot F, M. **Không có cột missing** vì không thiếu giá trị nào (quy tắc đã chốt: cột missing chỉ khi field có thiếu). Giá trị lạ → báo lỗi, như XuetangX. |
| tuổi | age = năm bắt đầu khóa − năm sinh, là số, z-score | OULAD **không có năm sinh**, chỉ có nhóm tuổi `age_band` (3 nhóm). Xem câu hỏi Q1. |

`disability` vẫn để ngoài X (em chưa yêu cầu).

### Q1. Mã hóa age_band thế nào? → **Em chọn A (2026-10-09)**

- **A (thầy đề xuất): one-hot 3 cột** `age_0-35`, `age_35-55`, `age_55<=`. Lý do: age_band là nhóm, không phải tuổi thật; giống cách đang làm với `imd_band` (cũng là nhóm) trong cùng pipeline; không giả định khoảng cách giữa các nhóm bằng nhau.
- B: số thứ tự 0 / 1 / 2 rồi z-score. Giống XuetangX hơn về hình thức (tuổi là 1 cột số, z-score), nhưng giả định 0-35 → 35-55 cách đều 35-55 → 55<=.

## 3. X mới

User block thêm gender rồi age_band ở đầu (như XuetangX: gender đứng đầu user block):

`user = gender (2) | age_band (3 nếu A, 1 nếu B) | region (13) | education (5) | imd (11) | credits, prev_attempts, registration (3)`

| | Hiện tại | A | B |
|---|---|---|---|
| user | 32 | 37 | 35 |
| X (full) | 103 | 108 | 106 |
| F1 (hành vi) | 60 | 60 | 60 |
| F2 (hành vi + user) | 92 | 97 | 95 |
| F3 (hành vi + course) | 71 | 71 | 71 |

Scaling: one-hot giữ 0/1; nếu B thì cột age vào `Z_COLUMNS` (z theo train).

## 4. File sửa

1. `0_config.py`: thêm `GENDERS = ("F", "M")`, `AGE_BANDS = ("0-35", "35-55", "55<=")`, chỉ số `GENDER_FEATURE_START`, `AGE_FEATURE_START`, dời chỉ số region/education/imd/số; sửa comment "gender, age_band and disability are protected" → chỉ còn disability.
2. `3_features.py`: thêm 2 + 3 dòng vào `feature_metadata()`, thêm 2 lệnh `set_one_hot(...)`, sửa comment chỉ số `[0, 60) behavior | [60, 97) user | [97, 108) course`.

Không sửa 2, 4–10 (hypergraph không dùng X; MLP/HGNN lấy số cột từ X.npy).

## 5. Kiểm tra sau khi code

- X có 108 cột, hữu hạn; mỗi node có đúng một 1 trong khối gender và một 1 trong khối age_band.
- Các cột cũ (hành vi, region…, course) giống hệt X cũ sau khi dời chỉ số.
- `feature_names.csv` khớp số cột.
- Smoke 10 epoch kịch bản M và F2.

Chạy lại từ `3_features.py` → `9_train.py`.
