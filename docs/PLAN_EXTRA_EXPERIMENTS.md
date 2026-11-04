# Kế hoạch thí nghiệm bổ sung: MLP-only, Shuffled incidence, một họ, số lớp HGNN (v2)

Ngày: 2026-11-04. Chạy trên cả OULAD và XuetangX, 5 seed (1, 11, 111, 1111, 11111), cùng split
60/10/30, cùng siêu tham số với M (lr 1e-3, wd 1e-5, dropout 0.3, 1000 epoch, patience 200, chọn
checkpoint theo val AUC).

Thay đổi so với v1, theo câu trả lời của em:
- MLP-only là P0: bỏ hẳn nhánh HGNN, chỉ giữ nhánh MLP. Không chạy X1.
- Thêm Object-only (O1).
- S1 chỉ hoán vị cơ bản, không sửa cặp trùng.
- Ghi tiếp kết quả vào `outputs/ABLATION` (đợt trước đã xong), không chạy lại M.

## 1. Các kịch bản

| Mã | Câu hỏi | Nhánh HGNN (họ siêu cạnh) | Lớp HGNN | MLP | Xáo | Ưu tiên |
|----|---------|---------------------------|----------|-----|-----|---------|
| M  | mô hình chính (đã có trong `outputs/ABLATION`) | C + O + U + S | 2 | ✓ | – | – |
| P0 | **MLP-only**: hypergraph có hơn đặc trưng riêng của node không? | không có nhánh HGNN | – | ✓ | – | 1 |
| S1 | **Shuffled incidence**: cấu trúc siêu cạnh có mang thông tin không? | C + O + U + S | 2 | ✓ | ✓ | 1 |
| C1 | Course-only | C + S | 2 | ✓ | – | 2 |
| O1 | Object-only | O + S | 2 | ✓ | – | 2 |
| U1 | User-only | U + S | 2 | ✓ | – | 2 |
| L1 | 1 lớp HGNN (đã có) | C + O + U + S | 1 | ✓ | – | 2 |
| L3 | 3 lớp HGNN | C + O + U + S | 3 | ✓ | – | 2 |

C = Course, O = Object, U = User, S = Self-loop.

### P0 = MLP-only
Mô hình chỉ gồm nhánh MLP ở `7_mlp.py` và bộ phân loại tuyến tính:
$\mathrm{logit} = z_s \cdot u + b$, với $z_s = \mathrm{ReLU}(\mathrm{Dropout}(\mathrm{ReLU}(xA_1+c_1))A_2+c_2)$.
Không có $H$ và không có $W$. Khung huấn luyện giữ nguyên như M: full-batch, cùng optimizer, cùng early
stopping. Trong bảng tổng hợp, mọi cột họ siêu cạnh của P0 hiện ✗ và `hgnn_layers = 0`.

Lưu ý khi đọc kết quả: P0 có khoảng một nửa số tham số của M, vì không có nhánh HGNN. Vì vậy M − P0 đo
giá trị của cả nhánh hypergraph, gồm cả cấu trúc lẫn phần tham số thêm vào. Phần riêng của cấu trúc được
đo bằng M − S1 (cùng số tham số).

### S1 = Shuffled incidence (hoán vị cơ bản)
Trong từng họ (Course, Object, User), danh sách thành viên `node_ids` được hoán vị ngẫu nhiên bằng
`np.random.default_rng(seed)`. Siêu cạnh Self-loop giữ nguyên. Đồ thị được xáo một lần trên toàn bộ node
(train + val + test) cho mỗi seed và dùng chung cho mọi epoch, nên 5 seed cho ra 5 đồ thị khác nhau.

Cách xáo này giữ nguyên kích thước $\delta(e)$ của mọi siêu cạnh và số membership của mỗi node trong từng
họ. Vì vậy $d(v)$ và hệ số $w_S/d(v)$ giống hệt M. Chỉ "node nào đi cùng node nào" thay đổi.

Cặp trùng được giữ nguyên, không sửa. Trường hợp này chỉ xảy ra ở Object: XuetangX có 16.573 trên
2.783.160 membership (0,6%), OULAD có 14.449 trên 734.714 (2,0%), đo ở seed 1. Khi tạo $H$,
`coalesce()` cộng hai membership trùng thành $h(v,e)=2$, còn `edge_size` vẫn đếm 2. Do đó $d(v)$ và
$\delta(e)$ vẫn đúng bằng bản gốc; node đó chỉ được tính hai lần trong cùng siêu cạnh.

### C1, O1, U1: chỉ một họ
Mỗi kịch bản giữ Self-loop và MLP, chỉ để lại một họ siêu cạnh. Ba kịch bản này bổ sung cho A1–A3
(bỏ từng họ).

### L1, M, L3: số lớp HGNN
M có 2 lớp. L1 đã có sẵn. L3 thêm lớp thứ ba sau lớp 2: Dropout → Linear(128, 128) → propagate → ReLU.

## 2. Thay đổi code (src và bản sao `baseline/new_dataset/OULAD`)

Hiện các file 6–10 của hai bản giống hệt nhau và phần `SCENARIOS` trong `0_config.py` cũng giống nhau.
Thầy sửa trong src rồi chép sang bản OULAD. Riêng `5_graph_data.py` sửa riêng từng bản, vì bản OULAD
dùng `x55_path`.

1. `0_config.py`: `scenario()` nhận thêm `use_hgnn=True` và `shuffle=False`. Thêm các kịch bản
   - `P0`: `families=()`, `hgnn_layers=0`, `use_hgnn=False`, `learn_w=False`
   - `S1`: `shuffle=True`
   - `C1`: `("course", "self_loop")`
   - `O1`: `("object", "self_loop")`
   - `U1`: `("user", "self_loop")`
   - `L3`: `hgnn_layers=3`

   `SCENARIOS_ALL` giữ nguyên.
2. `5_graph_data.py`: thêm hàm `shuffle_members(graph, seed)`, hoán vị `node_ids` trong từng họ.
   `apply_scenario(data, scenario, seed)` gọi hàm này khi `scenario["shuffle"]` bật, trước bước thêm
   Self-loop.
3. `6_hgnn.py`: cho phép `layers` bằng 1, 2 hoặc 3. Chỉ khi `layers == 3` mới tạo `layer3`.
4. `8_model.py`: `DropoutModel` nhận thêm `use_hgnn`. Khi `use_hgnn=False` thì không tạo HGNN, đặt
   `logit = z_s·u + b`, và `weight_summary` trả về rỗng.
5. `9_train.py`: truyền `seed` vào `apply_scenario` và `use_hgnn` vào `DropoutModel`. Chỉ lấy
   `family_logits` khi có nhánh HGNN. **Không** thêm cột vào `results.csv`, để header của file cũ
   không đổi.
6. `scripts/run_ablation.sh`: `SCENARIOS="${SCENARIOS:-M H A1 A2 A3 A4 W1}"`.

Không đổi: `7_mlp.py`, `10_summary.py`.

Vì mọi lớp mới chỉ được tạo trong kịch bản mới, M và các kịch bản cũ có thứ tự khởi tạo tham số không đổi.
Kết quả cũ trong `outputs/ABLATION` vẫn dùng chung bảng được.

## 3. Kiểm tra ở máy trước khi chạy server

1. M, seed 1, 5 epoch: val/test AUC trước và sau khi sửa code giống hệt nhau, trên cả hai dataset.
2. S1, seed 1: mọi $\delta(e)$ bằng bản gốc; $d(v)$ của mọi node bằng bản gốc (với $W$ ban đầu); in ra
   số cặp trùng và tỉ lệ membership bị đổi node; seed 11 cho ra đồ thị khác seed 1.
3. P0: mô hình không có tham số HGNN; số tham số bằng MLP cộng bộ phân loại.
4. Smoke 2 epoch, seed 1, cho P0 S1 C1 O1 U1 L1 L3 trên cả hai dataset; `10_summary.py` chạy được
   trên `results.csv` gộp cả cũ lẫn mới.

## 4. Lệnh chạy trên server

```bash
# Đợt 1 (bắt buộc)
SCENARIOS="P0 S1" bash scripts/run_ablation.sh
# Đợt 2 (nếu còn thời gian), sau khi đợt 1 xong
SCENARIOS="C1 O1 U1 L1 L3" bash scripts/run_ablation.sh
```

Mỗi đợt chạy OULAD trước, XuetangX sau, ghi tiếp vào `outputs/ABLATION/<dataset>/results.csv`, rồi tự gọi
`10_summary.py`. Ước tính thời gian dựa trên M ở XuetangX (khoảng 31 phút/seed): đợt 1 mất khoảng 3–5 giờ
(P0 nhanh hơn nhiều vì không có H), đợt 2 mất khoảng 12–14 giờ (L3 chậm hơn). OULAD nhanh hơn nhiều.

## 5. Cách đọc kết quả

Mọi so sánh đều là chênh lệch test AUC theo **từng seed** (cùng seed thì cùng split), báo mean ± std
trên 5 seed. Mức nhiễu đã thấy là khoảng ±0.0003–0.0009.

| So sánh | Ý nghĩa |
|---------|---------|
| M − P0 | giá trị của cả nhánh hypergraph so với chỉ đặc trưng riêng của node |
| M − S1 | giá trị của cấu trúc (ai đi cùng ai), khi bậc, kích thước và số tham số đã giữ nguyên |
| S1 − P0 | ≈ 0 hoặc < 0: đồ thị xáo không có ích hoặc còn gây hại |
| C1, O1, U1 so với P0 và M | lợi ích của từng họ khi đứng một mình |
| L1, M, L3 | độ nhạy theo số lớp |

Kết luận "cấu trúc hypergraph mang thông tin" cần cả hai điều kiện: M > P0 và M > S1, với chênh lệch vượt
mức nhiễu trên cả hai dataset.
