# Kịch bản thực nghiệm, bản 4

Ngày lập: 2026-10-06 (bản 3: 2026-10-04). **Đã code ngày 2026-10-06**: kịch bản nằm trong `SCENARIOS` của
`src/0_config.py`, chạy bằng `python src/9_train.py --scenario all --seeds 1 11 111 1111 11111`. X1 có trong
code nhưng không thuộc `all`. Kế hoạch của bản 4: `docs/KICH_BAN_THUC_NGHIEM_v4.md`.

**Thay đổi so với bản 3:**
- Kịch bản bỏ MLP đổi mã từ **B1** thành **H** (chỉ còn hypergraph): xem mô hình đầy đủ cho kết quả thế nào khi thiếu MLP.
- Thứ tự mới: M, A1–A4, W1, F1–F3, L1, H.
- Mọi phép bỏ/đổi khác giữ nguyên, vẫn làm trên mô hình đầy đủ M.

---

## 1. Mô hình gốc (M)

Mỗi kịch bản chỉ đổi **đúng một yếu tố** so với M.

| Yếu tố | Giá trị của M |
|---|---|
| Hyperedge | Course ✓, Object ✓, User ("any") ✓, self-loop ✓ |
| Học W | có, một trọng số cho mỗi loại hyperedge |
| Feature | `full`: hành vi + người học + khóa học (89 cột) |
| Nhánh HGNN | 2 layer |
| Nhánh MLP | có, 2 layer |
| Đầu ra | `logit = [z_g ‖ z_s]·u + b` |
| Train | tối đa 1000 epoch, early stopping theo val AUC (mỗi 5 epoch, patience 40) |
| Đánh giá | val và test ở ngưỡng 0.5 |
| Seed | 1, 11, 111, 1111, 11111 |

Các nhóm feature (hành vi được scale bằng log1p + z-score, age bằng z-score, one-hot giữ 0/1):

| Mã | Cột của X | Số cột |
|---|---|---|
| `feature` | hành vi: 35 ngày + tổng sự kiện + số object + 22 action (như MST-GCN) | 59 |
| `feature+user` | hành vi + giới tính (3) + học vấn (8) + tuổi (1) | 71 |
| `feature+course` | hành vi + lĩnh vực khóa học (18) | 77 |
| `full` | hành vi + người học + khóa học | 89 |

## 2. Bảng kịch bản (11 cấu hình × 5 seed = 55 lần chạy)

✓ = giữ, ✗ = bỏ. Ô **in đậm** là yếu tố khác với M.

| # | Mã | Course | Object | User | Self-loop | Học W | Feature | Layer HGNN | MLP | Câu hỏi trả lời |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **M** | ✓ | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ | Mô hình đầy đủ |
| 2 | A1 | **✗** | ✓ | ✓ | ✓ | ✓ | full | 2 | ✓ | Course đóng góp bao nhiêu |
| 3 | A2 | ✓ | **✗** | ✓ | ✓ | ✓ | full | 2 | ✓ | Object đóng góp bao nhiêu |
| 4 | A3 | ✓ | ✓ | **✗** | ✓ | ✓ | full | 2 | ✓ | User đóng góp bao nhiêu |
| 5 | A4 | ✓ | ✓ | ✓ | **✗** | ✓ | full | 2 | ✓ | Self-loop đóng góp bao nhiêu |
| 6 | W1 | ✓ | ✓ | ✓ | ✓ | **✗ (W = I)** | full | 2 | ✓ | Lợi ích của việc học W |
| 7 | F1 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature** | 2 | ✓ | Chỉ có feature hành vi |
| 8 | F2 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature+user** | 2 | ✓ | Hành vi + người học |
| 9 | F3 | ✓ | ✓ | ✓ | ✓ | ✓ | **feature+course** | 2 | ✓ | Hành vi + khóa học |
| 10 | L1 | ✓ | ✓ | ✓ | ✓ | ✓ | full | **1** | ✓ | HGNN 1 layer so với 2 layer |
| 11 | H | ✓ | ✓ | ✓ | ✓ | ✓ | full | 2 | **✗** | Mô hình đầy đủ khi thiếu MLP |

Ghi chú:
- **A4** (bỏ self-loop):
  - đặc trưng của chính node chỉ vào nhánh HGNN qua các hyperedge mà node là thành viên;
  - mọi node đều có hyperedge Course, nên bậc d(v) > 0 và công thức không bị chia cho 0;
  - nhánh MLP vẫn giữ đặc trưng riêng của node.
- **W1:** mọi hyperedge có `w_e = 1` trong suốt quá trình train (`G = Dv^-1/2·H·De^-1·Hᵀ·Dv^-1/2`, đúng như code gốc iMoonLab/HGNN). W1 có ít hơn M đúng 4 tham số (θ_course, θ_object, θ_user, θ_self_loop). Các cột `w_*` trong kết quả để trống.
- **F1–F3:** nhóm feature được áp cho cả hai nhánh HGNN và MLP, vì hai nhánh đọc cùng một X.
- **L1:** chỉ nhánh HGNN còn 1 layer `Z_g = ReLU(G·(X·Θ1 + b1))`; nhánh MLP giữ 2 layer.
- **H:** `logit = Z_g·u + b`, chỉ đọc nhánh HGNN (bản 3 gọi là B1).

## 3. Cách đọc kết quả

| So sánh | Đọc ra |
|---|---|
| M − A1, M − A2, M − A3 | Đóng góp của Course, Object, User khi đã có các loại còn lại |
| M − A4 | Đóng góp của self-loop trong nhánh HGNN |
| M − W1 | Lợi ích của việc học trọng số W theo loại hyperedge, so với W = I |
| F1 → F2, F1 → F3, F1 → M | Đóng góp của feature người học, feature khóa học, và cả hai |
| M − L1 | Lợi ích của bước nhảy thứ 2 |
| M − H | Đóng góp của nhánh MLP |

Chỉ coi một chênh lệch là thật khi nó lớn hơn khoảng **2 lần độ lệch chuẩn giữa các seed** (khoảng 0.0002–0.0005 AUC).

## 4. Tùy chọn (mặc định KHÔNG chạy)

| Mã | Course | Object | User | Self-loop | Học W | Feature | Layer HGNN | MLP | Câu hỏi trả lời |
|---|---|---|---|---|---|---|---|---|---|
| X1 | ✗ | ✗ | ✗ | ✓ | ✓ | full | 2 | ✓ | Đối chứng không có hàng xóm: G = I, nhánh HGNN thành MLP. Đo đóng góp của **toàn bộ** graph |

Chạy bằng `--scenario X1`. X1 không thuộc `all`.

## 5. Chạy

```
python src/9_train.py --scenario all --seeds 1 11 111 1111 11111
bash scripts/run_tmux.sh --scenario all --seeds 1 11 111 1111 11111     # trên server
python src/10_summary.py                                                # bảng mean ± std
```

- `all` gồm 11 kịch bản ở mục 2, chạy theo đúng thứ tự trong bảng. Mỗi kịch bản chạy hết 5 seed rồi mới sang kịch bản tiếp theo.
- **M đã có 5 seed** trên server (0.8759 ± 0.0001, commit 11db9ec). Chỉ chạy 10 kịch bản còn lại:
  `--scenario A1 A2 A3 A4 W1 F1 F2 F3 L1 H` (khoảng 4.7 phút mỗi lần → khoảng 4 giờ cho 50 lần).
- Mỗi lần chạy xong ghi ngay 1 dòng vào `outputs/results.csv`. Nếu bị dừng giữa chừng, các dòng đã ghi vẫn còn.
- Chạy thử nhanh: `--scenario all --seeds 1 --epochs 10 --eval-limit 2000`.
- Bảng tổng hợp (`summary.csv`, `ket_qua.xlsx`) có mỗi kịch bản một dòng theo thứ tự ở mục 2, với các cột: mã, các cột ✓/✗, mean ± std của val/test, `best_epoch`, tỉ lệ w, và **Δ test AUC so với M**.
- Các dòng `B1` cũ trong `results.csv` (nếu có) không còn mã trong `SCENARIOS`, nên được xếp cuối bảng tổng hợp; chúng thuộc X cũ, không dùng.
