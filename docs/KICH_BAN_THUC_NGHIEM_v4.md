# Kịch bản thực nghiệm, bản 4

Ngày lập: 2026-10-06 (sửa lần 3). **Đã code ngày 2026-10-06** (bảng chính: `docs/KICH_BAN_THUC_NGHIEM.md`).

**Thay đổi so với bản 3** (`docs/KICH_BAN_THUC_NGHIEM.md`):
- Mọi phép bỏ/đổi vẫn làm trên mô hình đầy đủ **M** (có MLP), giữ nguyên mã cũ.
- Kịch bản bỏ MLP đổi mã từ **B1** thành **H** (chỉ còn hypergraph). Mục đích: xem mô hình đầy đủ cho kết quả thế nào khi thiếu MLP.
- Feature, cách scale (log1p + z-score), protocol train/đánh giá và seed **giữ nguyên**.

---

## 1. Mô hình gốc M

| Yếu tố | Giá trị của M |
|---|---|
| Hyperedge | Course ✓, Object ✓, User ✓, self-loop ✓ |
| Học W | có |
| Feature | `full` (89 cột) |
| HGNN | 2 layer |
| MLP | có (2 layer) |
| Đầu ra | `logit = [z_g ‖ z_s]·u + b` |

Mỗi kịch bản chỉ đổi **đúng một yếu tố** so với M.

## 2. Danh sách kịch bản (11 cấu hình × 5 seed = 55 lần chạy)

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
- Nhóm feature: `feature` 59 cột, `feature+user` 71, `feature+course` 77, `full` 89.
- **A4**: đặc trưng của chính node vào nhánh HGNN qua các hyperedge Course / Object / User chứa nó. Mọi node đều thuộc một hyperedge Course nên d(v) > 0. Nhánh MLP vẫn giữ đặc trưng riêng của node.
- **W1**: `w_e = 1` cho mọi hyperedge, các cột `w_*` để trống.
- **H**: `logit = z_g·u + b`, chỉ đọc nhánh HGNN (giống hệt B1 cũ, chỉ đổi mã).
- X1 (chỉ self-loop) vẫn là tùy chọn, không thuộc `all`.

## 3. Cách đọc kết quả

| So sánh | Đọc ra |
|---|---|
| M − A1, M − A2, M − A3 | Đóng góp của Course, Object, User |
| M − A4 | Đóng góp của self-loop |
| M − W1 | Lợi ích của việc học W, so với W = I |
| F1 → F2, F1 → F3, F1 → M | Đóng góp của feature người học, feature khóa học, và cả hai |
| M − L1 | Lợi ích của bước nhảy thứ 2 |
| M − H | Đóng góp của nhánh MLP |

Chỉ coi một chênh lệch là thật khi nó lớn hơn khoảng **2 lần độ lệch chuẩn giữa các seed** (khoảng 0.0002–0.0005 AUC).

## 4. Chạy

```
python src/9_train.py --scenario all --seeds 1 11 111 1111 11111     # 11 kịch bản, theo thứ tự mục 2
python src/10_summary.py
```

- **M đã có 5 seed** trên server (0.8759 ± 0.0001, commit 11db9ec). Code từ đó tới nay chỉ sửa comment, nên có thể giữ 5 dòng này và chạy `--scenario A1 A2 A3 A4 W1 F1 F2 F3 L1 H`.
- Thời gian: khoảng 4.7 phút mỗi lần trên server → 50 lần còn lại khoảng **4 giờ**.
- Bảng tổng hợp giữ nguyên: mỗi kịch bản một dòng theo thứ tự ở mục 2, cột `delta_test_auc_vs_M`.

## 5. Phần code cần sửa

| File | Sửa |
|---|---|
| `0_config.py` | đổi mã `B1` thành `H` (mô tả: "M bỏ nhánh MLP, chỉ còn hypergraph"); xếp lại `SCENARIOS` và `SCENARIOS_ALL` theo thứ tự mục 2 (W1 lên sau A4, H xuống cuối) |
| `9_train.py`, `8_model.py` | sửa comment nhắc tới "B1" thành "H" |
| `docs/KICH_BAN_THUC_NGHIEM.md` | cập nhật theo bản 4 |

Không đổi phần tính toán nào, `10_summary.py` giữ nguyên.

Kiểm tra sau khi code: chạy thử nhanh `--scenario all --seeds 1 --epochs 10 --eval-limit 2000` trong scratchpad, phải ra đủ 11 dòng với mã H ở cuối.
