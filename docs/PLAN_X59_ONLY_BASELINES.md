# Plan: đầu vào của mọi baseline = X59 (bản 2)

Ngày: 2026-10-18. Em đã đồng ý; đã code và kiểm tra (mục 4) trên seed 1.

## 1. Nguyên tắc

1. Mỗi enrollment chỉ mang **X59** (`x59_s{seed}.npz`, cột `X`), đúng hàng của seed, không thêm cột.
2. Giữ **kiến trúc và siêu tham số của tác giả**; chỉ sửa chỗ bắt buộc để đọc X59.
3. Không dùng nhân khẩu học, thông tin khóa học, hay đặc trưng gom sẵn theo user / course / object.
   Node user / course / object của mô hình graph **không có đặc trưng riêng** (= 0, giữ số chiều), như
   hyperedge của M; mô hình vẫn dùng cấu trúc graph của nó và tự gom X59 bên trong.

## 2. Kiểm tra hiện tại: X59 đi vào đâu

| Baseline | X59 vào ở đâu | Đúng hàng / đúng seed | Thông tin ngoài X59 |
|---|---|---|---|
| LR, SVM, XGBoost | `load_splits`: `data["X"][rows]` | có | không → **giữ nguyên** |
| CFIN | `load_data`: `data['X'][rows]`, index = enroll_id | có | `feat_augment`: 4 × 59 cột mean / max theo user, theo course |
| CA-TFHN | `org_context` = X59, `seq_feat` = X59[:, 0:35] (đã so từng hàng: trùng) | có | node user / course ở link prediction = mean X59 theo user / course |
| MST-GCN | `apply_split`: `data['X'][rows]` theo enroll_id | có | node course: 4 cột (total_actions, num_enrollments, num_users, density); node object: 22 cột phân bố hành động |

## 3. Nội dung sửa từng baseline

### 3.1 LR, SVM, XGBoost

Không sửa.

### 3.2 CFIN (`dropout_prediction/main.py`, qua `cfin_changes.patch`)

model.py đọc mỗi đặc trưng thành nhóm 5 field (`conv1d` kernel 5, stride 5, model.py:78; attention trên
`a_field_size // 5` nhóm), nên giữ 295 field, 4 field gom = 0. model.py **không đổi**.

Thay hàm `feat_augment` (xoá) bằng:

```python
# changed: X59 only. model.py reads every feature as a group of 5 fields (conv kernel 5, stride 5):
# [x_j, user mean, user max, course mean, course max] in the original; here the 4 user / course
# fields are 0, so every group is [x_j, 0, 0, 0, 0] and the model sees X59 alone.
def feat_x59_only(train_feat, valid_feat, test_feat):
    act_feat = []
    for f in act_feat_basic:
        act_feat.append(f)
        for suffix in ['#user#mean', '#user#max', '#course#mean', '#course#max']:
            for df in [train_feat, valid_feat, test_feat]:
                df[f + suffix] = 0.0
            act_feat.append(f + suffix)
    return train_feat, valid_feat, test_feat, act_feat
```

Lời gọi: `dfTrain_a, dfValid_a, dfTest_a, act_feat = feat_x59_only(frames[0], frames[1], frames[2])`.
Sửa comment đầu file (dòng "feat_augment is kept ..."). `params` giữ nguyên.

### 3.3 CA-TFHN (`src/linkprediction/linkprediction_models.py`, qua `catfhn_changes.patch`)

Đặc trưng node user / course = 0, **trả lại số chiều của tác giả** (`user_features_len` 5,
`course_features_len` 2), vì toàn 0 thì số chiều không mang thông tin. `EncoderModel` vẫn có embedding
ID `user_embed`, `course_embed`, nên link prediction học từ cấu trúc ai học khóa nào.

| Chỗ | Hiện tại | Sau khi sửa |
|---|---|---|
| dòng 37–44 | `user_feat = all_log.groupby('username', sort=False)[x_columns].mean()` | `unique_user_id = all_log['username'].unique()`; `user_feat = torch.zeros((len(unique_user_id), 5))` |
| dòng 50–52 | `course_feat = all_log.groupby('course_id', sort=False)[x_columns].mean()` | `unique_course_id = all_log['course_id'].unique()`; `course_feat = torch.zeros((len(unique_course_id), 2))` |
| dòng 38–40 | đọc `['username', 'course_id'] + x_columns` | chỉ đọc `['username', 'course_id']` |
| `param_dict` | `user_features_len` 59, `course_features_len` 59 | **5 và 2 như gốc** |
| comment dòng 32–36 | mean X59 | "X59 only: user / course nodes have no feature of their own (0, the authors' 5 / 2 columns)" |

`unique()` giữ thứ tự xuất hiện đầu tiên như `groupby(sort=False)` và `3_mapping.py`, nên id graph
không đổi. `train.py`, `1_graph_addition_information.py` không đổi (`org_context` = X59,
`seq_feat` = X59[:, 0:35]).

Chỉnh bắt buộc do X59 (giữ, ghi trong bài): `org_context_feat_len` 7 → 59, `activity_num` 22 → 1,
`lstm_input_features` 184 → 16 (X59 có 35 cột theo ngày, không có 35 × 22).

### 3.4 MST-GCN (`xuetangx_graph_builder.py`, qua `mstgcn_changes.patch`)

Số chiều node lấy từ dữ liệu (trainer.py:222), nên điền 0 đúng số chiều cũ thì model không đổi.

| Chỗ | Hiện tại | Sau khi sửa |
|---|---|---|
| (B) course | `course_behavior` = total_actions, num_enrollments, num_users, density → StandardScaler | xoá; `crs_vec = np.zeros((len(course_ids), 4))` |
| (C) object | `pd.crosstab(object, action)` → StandardScaler | xoá; `obj_vec = np.zeros((len(object_ids), logs['action'].nunique()))` (22) |
| `scaler = StandardScaler()` | dùng cho B, C | xoá (không còn dùng) |
| `num_course_feats` | `crs_vec_scaled.shape[1]` | `crs_vec.shape[1]` (= 4) |
| comment | | "X59 only: course and object nodes have no feature of their own (0, the original 4 / 22 columns)" |

Siêu tham số giữ như gốc (hidden 128, 2 lớp, lr 1e-4, batch 256, dropout 0.3, 8 epoch).

## 4. Kiểm tra sau khi code

1. CFIN: 295 field; với mọi j, cột x_j trùng X59, 4 cột còn lại = 0.
2. CA-TFHN: `user_feat`, `course_feat` toàn 0 (5 / 2 cột); `org_context` = X59, `seq_feat` = X59[:, 0:35]
   từng hàng; id user / course giống `graph_mapping.csv`.
3. MST-GCN: feature course (4) và object (22) toàn 0; enrollment = X59 theo enroll_id.
4. Mỗi patch áp được lên clone sạch; smoke từng baseline chạy hết.

## 5. Chạy trên server

```bash
git pull
bash baseline/CFIN/run_cfin.sh setup && bash baseline/CFIN/run_cfin.sh full
bash baseline/CA-TFHN/run_catfhn.sh setup && bash baseline/CA-TFHN/run_catfhn.sh features && bash baseline/CA-TFHN/run_catfhn.sh full
bash baseline/MST-GCN/run_mstgcn.sh setup && bash baseline/MST-GCN/run_mstgcn.sh features && bash baseline/MST-GCN/run_mstgcn.sh full
python baseline/summary.py
```

Kết quả cũ của ba baseline này chuyển vào `outputs/baselines/_old_with_aux/` trước khi chạy.
