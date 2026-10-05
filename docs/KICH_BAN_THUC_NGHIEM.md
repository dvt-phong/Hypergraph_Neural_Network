# Kịch bản thực nghiệm, bản 3

Ngày lập: 2026-10-04. **Đã code ngày 2026-10-04**: kịch bản nằm trong `SCENARIOS` của `src/0_config.py`, chạy bằng
`python src/9_train.py --scenario all --seeds 1 11 111 1111 11111`. X1 có trong code nhưng không thuộc `all`.

**Thay đổi so với bản 2:**
- Bảng viết lại dạng ✓ / ✗ cho từng thành phần, để thấy ngay kịch bản nào giữ gì, bỏ gì.
- **A4 = giữ Course, Object, User; bỏ self-loop** (ở bản 2 kịch bản này mang mã A5).
- Kịch bản "chỉ còn self-loop" là đề xuất của thầy, không phải yêu cầu của em. Nó được chuyển xuống mục 4 thành **tùy chọn X1** và mặc định **không chạy**.

---

## 1. Mô hình gốc (M)

Mỗi kịch bản chỉ đổi **đúng một yếu tố** so với M.

| Yếu tố | Giá trị của M |
|---|---|
| Hyperedge | Course ✓, Object ✓, User ("any") ✓, self-loop ✓ |
| Feature | `full`: hành vi + người học + khóa học (89 cột) |
| Nhánh HGNN | 2 layer |
| Nhánh MLP | có, 2 layer, cấu trúc không đổi |
| Train | tối đa 1000 epoch, early stopping theo val AUC (mỗi 5 epoch, patience 40) |
| Đánh giá | val và test ở ngưỡng 0.5 |
| Seed | 1, 11, 111, 1111, 11111 |

Các nhóm feature (từ 2026-10-05: hành vi như MST-GCN gồm 35 ngày để thô + tổng sự kiện, số object, 22 action chuẩn hóa; age theo CFIN; bỏ các cột `_other`):

| Mã | Cột của X | Số cột |
|---|---|---|
| `feature` | hành vi: 35 ngày + tổng sự kiện + số object + 22 action (như MST-GCN) | 59 |
| `feature+user` | hành vi + giới tính (3) + học vấn (8) + tuổi (1) | 71 |
| `feature+course` | hành vi + lĩnh vực khóa học (18) | 77 |
| `full` | hành vi + người học + khóa học | 89 |

Mỗi khối one-hot có một cột `missing` (thiếu là một mức riêng). Tuổi thiếu hoặc ngoài 10–70 được gán 0 rồi chuẩn hóa, như CFIN.

## 2. Bảng kịch bản (11 cấu hình × 5 seed = 55 lần chạy)

✓ = giữ, ✗ = bỏ. Ô **in đậm** là yếu tố khác với M.

| # | Mã | Course | Object | User | Self-loop | Feature | Layer HGNN | MLP | Học W | Câu hỏi trả lời |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **M** | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ | ✓ | Mô hình chính |
| 2 | A1 | **✗** | ✓ | ✓ | ✓ | full | 2 | ✓ | ✓ | Course đóng góp bao nhiêu |
| 3 | A2 | ✓ | **✗** | ✓ | ✓ | full | 2 | ✓ | ✓ | Object đóng góp bao nhiêu |
| 4 | A3 | ✓ | ✓ | **✗** | ✓ | full | 2 | ✓ | ✓ | User đóng góp bao nhiêu |
| 5 | A4 | ✓ | ✓ | ✓ | **✗** | full | 2 | ✓ | ✓ | Self-loop đóng góp bao nhiêu |
| 6 | F1 | ✓ | ✓ | ✓ | ✓ | **feature** | 2 | ✓ | ✓ | Chỉ có feature hành vi |
| 7 | F2 | ✓ | ✓ | ✓ | ✓ | **feature+user** | 2 | ✓ | ✓ | Hành vi + người học |
| 8 | F3 | ✓ | ✓ | ✓ | ✓ | **feature+course** | 2 | ✓ | ✓ | Hành vi + khóa học |
| 9 | L1 | ✓ | ✓ | ✓ | ✓ | full | **1** | ✓ | ✓ | HGNN 1 layer so với 2 layer |
| 10 | B1 | ✓ | ✓ | ✓ | ✓ | full | 2 | **✗** | ✓ | Bỏ nhánh MLP: chỉ còn HGNN |
| 11 | W1 | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ | **✗** | Không học W: W = I cố định, như HGNN gốc |

Ghi chú:
- **A4** (bỏ self-loop):
  - đặc trưng của chính node chỉ vào nhánh HGNN qua các hyperedge mà node là thành viên;
  - mọi node đều có hyperedge Course, nên bậc d(v) > 0 và công thức không bị chia cho 0;
  - nhánh MLP vẫn giữ đặc trưng riêng của node.
- **F1–F3:** nhóm feature được áp cho cả hai nhánh HGNN và MLP, vì hai nhánh đọc cùng một X.
- **L1:** chỉ nhánh HGNN còn 1 layer `Z_g = ReLU(G·(X·Θ1 + b1))`; nhánh MLP giữ 2 layer.
- **B1:** `logit = Z_g·u + b`, chỉ đọc nhánh HGNN.
- **W1:** mọi hyperedge có `w_e = 1` trong suốt quá trình train (`G = Dv^-1/2·H·De^-1·Hᵀ·Dv^-1/2`, đúng như code gốc iMoonLab/HGNN). W1 có ít hơn M đúng 4 tham số (θ_course, θ_object, θ_user, θ_self_loop). Các cột `w_*` trong kết quả để trống.

## 3. Cách đọc kết quả

| So sánh | Đọc ra |
|---|---|
| M − A1, M − A2, M − A3 | Đóng góp của Course, Object, User khi đã có các loại còn lại |
| M − A4 | Đóng góp của self-loop trong nhánh HGNN |
| F1 → F2, F1 → F3, F1 → M | Đóng góp của feature người học, feature khóa học, và cả hai |
| M − L1 | Lợi ích của bước nhảy thứ 2 |
| M − B1 | Đóng góp của nhánh MLP |
| M − W1 | Lợi ích của việc học trọng số W theo loại hyperedge, so với W = I |

Chỉ coi một chênh lệch là thật khi nó lớn hơn khoảng **2 lần độ lệch chuẩn giữa các seed**. Theo các lần chạy trước, mức này vào khoảng 0.0005 AUC.

## 4. Tùy chọn (mặc định KHÔNG chạy)

| Mã | Course | Object | User | Self-loop | Feature | Layer HGNN | MLP | Học W | Câu hỏi trả lời |
|---|---|---|---|---|---|---|---|---|---|
| X1 | ✗ | ✗ | ✗ | ✓ | full | 2 | ✓ | ✓ | Đối chứng không có hàng xóm: G = I, nhánh HGNN thành MLP. Đo đóng góp của **toàn bộ** graph |

Chạy bằng `--scenario X1`. X1 không thuộc `all`.

## 5. Chạy toàn bộ

```
python src/9_train.py --scenario all --seeds 1 11 111 1111 11111
bash scripts/run_tmux.sh --scenario all --seeds 1 11 111 1111 11111     # trên server
python src/10_summary.py                                                # bảng mean ± std
```

- `all` gồm 11 kịch bản ở mục 2, chạy theo đúng thứ tự trong bảng. Mỗi kịch bản chạy hết 5 seed rồi mới sang kịch bản tiếp theo.
- Mỗi lần chạy xong ghi ngay 1 dòng vào `outputs/results.csv`. Nếu bị dừng giữa chừng, các dòng đã ghi vẫn còn.
- Chạy một phần: `--scenario M A4 B1`. Chạy thử nhanh: `--scenario all --seeds 1 --epochs 10 --eval-limit 2000`.
- Bảng tổng hợp (`summary.csv`, `ket_qua.xlsx`) có mỗi kịch bản một dòng theo thứ tự ở mục 2, với các cột: mã, các cột ✓/✗, mean ± std của val/test, `best_epoch`, tỉ lệ w, và **Δ test AUC so với M**.
- Thời gian ước tính trên CPU local, trường hợp xấu nhất đủ 1000 epoch: khoảng 70 phút một lần chạy, tức khoảng 64 giờ cho 55 lần. Early stopping và GPU sẽ rút ngắn đáng kể (GPU thầy chưa đo).

## 6. Phần code (đã làm; W1 thêm `learn_w` vào `SCENARIOS`, `HGNNEncoder` và `DropoutModel`)

| File | Thêm |
|---|---|
| `0_config.py` | bảng `SCENARIOS`: mỗi mã gồm `families` (tập con của course, object, user, self_loop), `features`, `hgnn_layers`, `use_mlp`, và 1 câu mô tả |
| `3_features.py` | `feature_columns(name)`: trả về các cột X ứng với `feature`, `feature+user`, `feature+course`, `full` |
| `5_graph_data.py` | chỉ giữ hyperedge thuộc `families`; thêm self-loop chỉ khi có `self_loop`; với target, bỏ hyperedge của các loại không dùng |
| `6_hgnn.py` | tham số `layers` (1/2) và `use_self_loop`, áp cho cả forward lúc train và Bước A/B lúc đánh giá |
| `8_model.py` | tham số `use_mlp`: khi False thì classifier là `Linear(hidden, 1)` |
| `9_train.py` | `--scenario all` hoặc danh sách mã; ghi thêm cột `scenario, families, features, hgnn_layers, use_mlp` vào `results.csv` |
| `10_summary.py` | gom theo `scenario` theo thứ tự bảng, thêm các cột ✓/✗ và Δ test AUC so với M |

Kiểm tra sau khi code:
- Lặp lại 2 phép thử nhất quán (node train qua công thức đánh giá trùng forward đầy đủ; vòng lặp theo định nghĩa trên 50 target) cho các trường hợp đặc biệt **A4, L1, B1**.
- Chạy thử nhanh `--scenario all --seeds 1 --epochs 10 --eval-limit 2000`: `results.csv` phải có đủ 10 dòng.

## 7. Đã chốt

- 11 kịch bản ở mục 2 (`--scenario all`), mỗi kịch bản chỉ đổi một yếu tố so với M.
- A4 = giữ Course, Object, User; bỏ self-loop; giữ MLP.
- B1 = bỏ MLP trên M.
- W1 = không học W (W = I) trên M.
- X1 chỉ chạy khi gọi tên.
