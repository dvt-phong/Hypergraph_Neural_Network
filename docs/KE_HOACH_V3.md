# Kế hoạch tổng thể (bản 3)

Ngày lập: 01/10/2026. Bản này gom lại những gì đã chạy (lần 2, P0, bước A, bước B,
C0–C2), phần tự phản biện, và hướng mới: **giữ nguyên mô hình đề xuất ban đầu**
(HSL + contrastive, sơ đồ [v3](assets/hypergraph-neural-network-v3.png)), bổ sung
**ma trận W học được** và **hyperedge User**. Chi tiết từng thí nghiệm cũ nằm ở
[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md); phân tích nguyên nhân nằm ở
[PHAN_TICH_HSL.md](PHAN_TICH_HSL.md).

---

## 1. Đã biết gì (test, trừ khi ghi khác)

| Mô hình | Seed | Val AUC | Test AUC | Ghi chú |
|---|---|---|---|---|
| HSL ban đầu (lần 2) | 5 | – | 0,8092 ± 0,0029 | F1 0,45 ± 0,31 do `pos_weight` và ngưỡng 0,5 |
| HSL, sửa giao thức (P0) | 5 | – | 0,8324 ± 0,0009 | F1 0,897 ± 0,0005 |
| HSL, train 1000 epoch (b0) | 1 | – | 0,8332 | Dừng sớm không phải nguyên nhân |
| HGNN (b0-hgnn) | 1 | – | 0,8467 | HSL thấp hơn HGNN 1,35 điểm |
| HSL + skip (b1) | 1 | – | 0,8699 | Skip +3,7 điểm |
| MLP | 5 | 0,8673 | 0,8693 ± 0,0001 | |
| GBDT | 5 | 0,8691 | 0,8699 ± 0,0002 | Chưa tinh chỉnh |
| HGNN + skip + W family, 3 loại (fw2-hgnn) | 5 | 0,8705 | 0,8723 ± 0,0012 | W tự tắt Behavioral (1–3% self-loop) |
| **HGNN + skip + W family, Course + Object (fw2-nob-hgnn)** | 5 | **0,8716** | **0,8737 ± 0,0002** | Mô hình chính hiện tại. DeLong hơn GBDT p ≤ 6·10⁻⁷ |

Năm điều đã chắc chắn:

1. **Giao thức** (`pos_weight` 1, t\* trên validation, train đủ lâu) là sửa lỗi, không
   phải lựa chọn.
2. **Pha loãng đặc trưng của chính node** là nguyên nhân lớn nhất: sau 2 lớp chỉ còn
   khoảng 1,5%. Skip hoặc trọng số self-loop cao đều sửa được.
3. **Behavioral (kNN trên X) không thêm thông tin**, và gây lệch giữa lúc train và lúc
   đánh giá. W học được tự đưa nó về khoảng 1%.
4. **Mask cứng của HSL không cắt được** (`kept` ≈ 0,99, gradient bão hoà). Mask này
   chỉ có lợi khi graph còn hyperedge có hại.
5. **Course, Object và User có thông tin** ngoài X. Đo bằng GBDT + trung bình
   hàng xóm (val): Course + Object +0,0054, thêm User (`any`) được thêm +0,0037.

Ba điều **chưa** biết:

- HSL có ích không khi đã có W, User và được tinh chỉnh công bằng (phạt độ thưa, giảm τ).
- W theo **từng hyperedge** có hơn W theo family không.
- HGNN có hơn một **GBDT được tinh chỉnh, dùng đặc trưng trung bình hàng xóm** không.

---

## 2. Mô hình đề xuất (bản 3)

Giữ nguyên toàn bộ khung trong sơ đồ v3. Chỉ thêm W vào **phép lan truyền**, dùng
chung cho cả Z0 và Z\*.

```text
H0 = Course + Object + Behavioral + User (+ self-loop)

Z0     = HGNN_W(X, H0)
H*     = Me ⊙ Mv ⊙ (H0 + ΔH) + I                (HSL, như cũ)
Z*     = HGNN_W(X, H*)                         (cùng encoder, cùng W)
ŷ      = σ(Linear(Z*))
L      = BCE + λ · L_CL(Z0, Z*)

HGNN_W: X' = Dv^-1/2 · H · W · De^-1 · Hᵀ · Dv^-1/2 · XΘ
        W_ee = w_f(e) · α_e
        w_f  = softplus(θ_f)                       một số cho mỗi loại hyperedge
        α_e  = 2σ(g([mean X của e ‖ one-hot loại ‖ log |e|]))    một số cho mỗi hyperedge
```

| Thành phần | Vai trò | Vì sao thiết kế như vậy |
|---|---|---|
| `w_f` (4 + 1 số) | Loại quan hệ nào quan trọng | Đã chứng minh ổn định qua 5 seed (Object/self ≈ 0,13) |
| `α_e` (MLP nhỏ, 32 chiều) | Khoá nào, video nào, nhóm học viên nào quan trọng | Đọc **X gốc**, không đọc Z0 đã bị làm mượt; không đọc nhãn; áp dụng được cho local graph chưa gặp |
| `α_e` khởi tạo = 1 | Lúc đầu mô hình giống hệt khi chưa có α | Không làm hỏng điểm xuất phát |
| `α_e` ∈ (0, 2) | Vừa tăng vừa giảm được | Khác mask HSL: gradient không bão hoà ở 0/1 |
| Self-loop: α = 1 | Chỉ có `w_self_loop` | Tránh việc hai tham số cùng làm một việc |
| lr riêng (w: 0,05; α: 0,005), không weight decay | | Bài học từ bản W cũ: lr 1e-3 và weight decay làm W gần như không đổi |
| User `any` | Mọi lượt đăng ký của cùng học viên | Kiểm tra độ vững bằng `temporal` (chỉ các khoá đã bắt đầu) |

**W và HSL khác nhau thế nào:** Me và Mv quyết định **giữ hay bỏ** (0/1, ngẫu nhiên
khi train), còn W quyết định **nghe nhiều hay ít** (liên tục, tất định). Hai cơ chế
bổ sung cho nhau: nếu HSL vẫn giữ gần hết (`kept` ≈ 0,99) thì W vẫn giảm được ảnh
hưởng của hyperedge xấu.

Sau khi train, `outputs/reports/<run>_edge_weights.csv` ghi cho mỗi hyperedge: loại,
khoá (mã khoá học, mã video, mã học viên), kích thước, `α_e`, `W_ee`, và tỉ lệ bỏ
học của thành viên. Nhãn chỉ dùng để **phân tích** bảng này, không bao giờ vào mô hình.

Code: [5_model.py](../src/5_model.py) (`edge_weights`), [8_train.py](../src/8_train.py)
(`--edge-weights`, `--edge-weight-lr`, `save_edge_weights`).

---

## 3. Lộ trình

```text
Giai đoạn 1  Sàng lọc seed 1: nhánh O (mô hình đề xuất) và nhánh U (HGNN + skip + User)
             ── Cổng 1: chọn cấu hình của mỗi nhánh bằng val AUC ──
Giai đoạn 2  5 seed cho cấu hình được chọn; DeLong
Giai đoạn 3  HSL công bằng: phạt độ thưa, giảm dần τ (chỉ nếu o3 ≈ o3-hgnn)
Giai đoạn 4  Baseline mạnh: GBDT tinh chỉnh + đặc trưng trung bình hàng xóm
             ── Cổng 2: chốt mô hình chính ──
Giai đoạn 5  Phân tích W (α_e), ablation, viết
```

Nguyên tắc (giữ từ bản 2): chọn bằng **val AUC**, test chỉ để báo cáo; sàng lọc 1
seed, chênh lệch val dưới 0,002 coi là nhiễu; cấu hình được chọn chạy 5 seed.

### Chạy Giai đoạn 1 và 2 một lần, qua đêm

```bash
git pull
RUN_SCRIPT=scripts/run_night.sh bash scripts/run_tmux.sh     # rồi Ctrl-b d để thoát, cứ để chạy
```

| Bước | Nội dung | Thời gian ước tính |
|---|---|---|
| 0 | Thêm User vào `hypergraph.npz` (bản cũ giữ ở `hypergraph_v3.npz`); dựng `hypergraph_temporal.npz` | vài phút |
| 1a | o3, o3-skip, o3-hgnn, o2, o1 (seed 1) | khoảng 6–7 giờ |
| 1b | u1-hgnn, u0-hgnn, u2-hgnn, u1-temporal (seed 1) | khoảng 3 giờ |
| 2 | Thêm seed 11, 111, 1111, 11111 cho o3, o3-skip, u1-hgnn | khoảng 14 giờ |

Nếu một cấu hình lỗi, các cấu hình khác vẫn chạy tiếp. Chạy lại một phần:
`STAGES="1b 2" RUN_SCRIPT=scripts/run_night.sh bash scripts/run_tmux.sh`.

Sau mỗi bước, `scripts/summarize_results.py` ghi **bảng kết quả chi tiết** cho mọi thư
mục trong `result/`:

| File | Nội dung |
|---|---|
| `result/tong_hop_ket_qua.xlsx`, sheet `Kich_ban` | Mỗi cấu hình một dòng (gộp seed 1 của bước 1 với 4 seed của bước 2), sắp theo test AUROC |
| sheet `Lan_chay` | Mỗi lần chạy (thư mục `result/<ngày_giờ>`) một dòng |
| sheet `Theo_seed` | Từng seed |
| sheet `Giai_thich` | Ý nghĩa từng cột |
| `result/tong_hop_ket_qua.md` | Bảng rút gọn các chỉ số test |

Các cột gồm: lần chạy, kịch bản (tag), mô hình, 24 siêu tham số, và cho cả val lẫn
test: AUROC (tức AUC), AUPRC, ACC, Precision, Recall, F1, Specificity, Macro-F1, F1 và
AUPRC lớp không bỏ học (mean ± std); ngưỡng t\*, best epoch, số epoch, thời gian,
`kept_*`, `w_*`, `alpha_*`. Mọi chỉ số được tính lại từ file xác suất đã lưu, nên các
lần chạy cũ cũng có ACC.

### Giai đoạn 1. Sàng lọc (seed 1, 1000 epoch, patience 60)

**Nhánh O: mô hình đề xuất + W + User** (`scripts/run_o.sh`, đã code):

| Tên | Cấu hình | Trả lời câu hỏi |
|---|---|---|
| o1 | HSL đầy đủ, 4 loại hyperedge | User có giúp mô hình gốc không (so với b0)? |
| o2 | o1 + W theo family | W family có sửa được pha loãng khi **không có skip**? |
| **o3** | o2 + W theo từng hyperedge | **Mô hình đề xuất.** α_e có thêm gì so với w_f? |
| o3-skip | o3 + skip | Có W rồi thì còn cần skip không? |
| o3-hgnn | o3 bỏ HSL | HSL có thêm gì khi đã có W? |

```bash
# Trên server, trong tmux. Script tự thêm User vào hypergraph.npz nếu chưa có
# (dùng lại kNN, giữ bản cũ ở hypergraph_v3.npz).
RUN_SCRIPT=scripts/run_o.sh bash scripts/run_tmux.sh
# Chạy thử một cấu hình trước để xem thời gian mỗi epoch:
ONLY="o3" RUN_SCRIPT=scripts/run_o.sh bash scripts/run_tmux.sh
```

**Nhánh U: mô hình chính + User** (`scripts/run_b.sh`, đã code từ trước, chưa chạy):

| Tên | Cấu hình |
|---|---|
| u0-hgnn | HGNN + skip, Course + Object + User |
| u1-hgnn | u0 + W family (mô hình chính + User) |
| u2-hgnn | HGNN + skip + W family, 4 loại |
| u1-temporal | u1 trên graph `temporal` |

```bash
python src/4_hypergraph.py --user-rule temporal --hypergraph-file hypergraph_temporal.npz \
       --reuse-neighbors hypergraph_v3.npz          # sau khi run_o.sh đã tạo hypergraph_v3.npz
ONLY="u0-hgnn u1-hgnn u2-hgnn u1-temporal" RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh
```

Thời gian ước tính: cấu hình HGNN khoảng 40 phút, cấu hình HSL khoảng 1–1,5 giờ
(hai lượt encode mỗi epoch). Chín cấu hình khoảng 8–10 giờ. Nếu chỉ có một GPU, chạy
nhánh O trước.

**Cần xem trong kết quả** (`results.csv`, `history.png`, `*_edge_weights.csv`):

| Cột | Đọc thế nào |
|---|---|
| `val_auc`, `test_auc` | So giữa các cấu hình; mốc là fw2-nob-hgnn seed 1 (val 0,8720) |
| `w_*` | Self-loop có tăng mạnh không (khi không có skip, đây là cách duy nhất để giữ đặc trưng của chính node)? Behavioral có bị tắt không? |
| `alpha_*` | Trung bình α theo loại. Gần 1 = α chưa học gì; tách xa 1 = đang phân biệt |
| `kept_*`, `added` | HSL có cắt không khi đã có W? |
| `train_auc` so với val | Còn underfit không (b0: train khoảng 0,85)? |

### Cổng 1

| Kết quả | Kết luận | Làm tiếp |
|---|---|---|
| o3 ≥ o3-skip − 0,002 | W thay được skip: **mô hình đề xuất giữ nguyên kiến trúc gốc** | Chọn o3 |
| o3 < o3-skip − 0,002 | Vẫn cần skip | Chọn o3-skip, ghi skip là một sửa đổi |
| o3 − o2 ≥ 0,002 | α_e có ích | Giữ α_e, phân tích bảng α |
| \|o3 − o2\| < 0,002 | α_e không thêm độ chính xác | Vẫn giữ nếu bảng α giải thích được (Giai đoạn 5); báo trung thực |
| o3 − o3-hgnn ≥ 0,002 | HSL có đóng góp khi có W | Bỏ qua Giai đoạn 3, chạy 5 seed |
| o3 ≤ o3-hgnn + 0,002 | HSL chưa đóng góp | Giai đoạn 3 |
| u1 − fw2-nob ≥ 0,002 | User có ích trong mô hình chính | 5 seed u1; so với u1-temporal |

### Giai đoạn 2. 5 seed và kiểm định

```bash
SEEDS="1 11 111 1111 11111" ONLY="<cấu hình O được chọn>" RUN_SCRIPT=scripts/run_o.sh bash scripts/run_tmux.sh
SEEDS="1 11 111 1111 11111" ONLY="u1-hgnn" RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh
python scripts/delong.py result/<O> result/<fw2-nob-hgnn> result/<GBDT> result/<MLP>
```

Đạt khi hơn mô hình chính (0,8737) có ý nghĩa theo DeLong ở cả 5 seed **và** mức hơn
lớn hơn 2 std giữa các seed. Nếu chỉ thoả DeLong: ghi "ngang bằng", vì DeLong không
tính dao động giữa các lần train.

### Giai đoạn 3. HSL công bằng (chỉ khi o3 ≈ o3-hgnn)

Làm trên cấu hình O được chọn, seed 1. Cần code (khoảng nửa ngày) trong
[6_hsl.py](../src/6_hsl.py) và [7_losses.py](../src/7_losses.py):

| Mã | Việc | Quét |
|---|---|---|
| C1 | Phạt độ thưa `β · mean(p_keep)` trên membership H0 | β ∈ {0,01; 0,1} |
| C2 | τ Gumbel giảm dần từ 1 xuống 0,1 | – |
| C3 | Khởi tạo xác suất giữ 73% thay vì 95% | – |
| C4 | Bỏ contrastive (`--lambda-cl 0`), chưa từng đo riêng | – |

Đạt khi `kept_*` ≤ 0,9 ổn định và val không giảm. Nếu không đạt: báo HSL mask cứng như
một **kết quả âm có phân tích** (gradient bão hoà, `kept` tăng dần), còn W là phần học
cấu trúc thay thế.

### Giai đoạn 4. Baseline mạnh (trả lời "cần HGNN để làm gì?")

Code `src/9_baselines.py --neighbor-means` (khoảng nửa ngày): thêm vào X trung bình đặc
trưng của hàng xóm theo từng loại (Course, Object, User, tính **đúng như local graph**:
target chỉ thấy thành viên train). Tinh chỉnh GBDT trên val (`max_iter`, `learning_rate`,
`max_leaf_nodes`), 5 seed.

| Kết quả | Cách viết |
|---|---|
| Mô hình đề xuất > GBDT + hàng xóm | Lan truyền nhiều lớp và W có giá trị ngoài phép trung bình đơn giản |
| Ngang nhau | Đóng góp chính chuyển sang **giải thích**: W và α_e cho biết quan hệ nào quan trọng, điều GBDT không cho |

### Cổng 2. Chốt mô hình chính

Chọn theo val AUC 5 seed giữa: cấu hình O, u1-hgnn và fw2-nob-hgnn. Nếu cấu hình O
ngang mô hình tốt nhất (trong 2 std), **ưu tiên cấu hình O**, vì nó là mô hình đề xuất
và giải thích được nhiều hơn.

### Giai đoạn 5. Phân tích và viết

1. **Bảng chính:** LR, GBDT, GBDT + hàng xóm, MLP, HGNN, HSL gốc, mô hình đề xuất. Các cột:
   AUC, AUPRC, F1 (t\*), macro-F1, AUPRC lớp không bỏ học, mean ± std 5 seed, DeLong.
2. **Ablation của mô hình đề xuất:** bỏ α_e, bỏ w_f, bỏ User, bỏ Behavioral, bỏ HSL,
   bỏ contrastive, `temporal` thay cho `any`.
3. **Phân tích W** (từ `*_edge_weights.csv`, cần viết script khoảng nửa ngày):
   - `w_f` qua 5 seed (bảng đã có cho bản family).
   - α_e so với kích thước và độ "thuần" của hyperedge (|tỉ lệ bỏ học − 0,5|).
   - **Kiểm tra độ trung thực:** bỏ 10% hyperedge có W cao nhất và 10% thấp nhất lúc
     đánh giá, so sánh AUC giảm bao nhiêu. Chỉ kết luận "W phản ánh độ hữu ích" khi bỏ
     nhóm cao làm AUC giảm rõ hơn.
   - Ví dụ cụ thể: 5 khoá học, 5 video có α cao nhất và thấp nhất.
4. **Kết quả âm của HSL mask cứng:** hình `kept_*` theo epoch, bảng gradient bão hoà.
5. Cập nhật [METHOD.md](METHOD.md) (mục 3.4: W; mục 3.3: User) và sơ đồ v4.

### Nếu còn thời gian

| Mã | Việc | Khi nào đáng làm |
|---|---|---|
| H7 | Train trên local graph, giống lúc đánh giá | Nếu giữ Behavioral và train/val AUC còn lệch |
| H4 | Thêm feature theo CFIN (session, recency, tỉ lệ làm đúng) | Tăng mọi mô hình, không đổi thứ hạng; làm cuối |
| Mức 3 | Attention theo node (HyperGAT/AllSet) | Nếu α_e có ích và cần giải thích theo từng học viên |

---

## 4. Lịch dự kiến

| Thời gian | Việc |
|---|---|
| Ngày 1–2 | Giai đoạn 1 (hai nhánh, khoảng 8–10 giờ GPU); đọc kết quả, Cổng 1 |
| Ngày 3–4 | Giai đoạn 2 (5 seed); code Giai đoạn 4 song song |
| Ngày 5–6 | Giai đoạn 3 nếu cần; Giai đoạn 4 chạy |
| Ngày 7 | Cổng 2: chốt mô hình |
| Tuần 2 | Giai đoạn 5: ablation, phân tích W, viết |

---

## 5. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Không có skip, mô hình đề xuất vẫn bị pha loãng (như b0: 0,833) | o2/o3 cho thấy W có tự tăng self-loop đủ không; nếu không, dùng o3-skip và ghi rõ |
| α_e không rời khỏi 1 | Xem `alpha_*` theo epoch; tăng `--edge-weight-lr` (0,02) cho một lần chạy thử |
| α_e học theo kích thước chứ không theo nội dung | Kiểm tra tương quan α với log \|e\|; báo trung thực |
| User `any` dùng hành vi của khoá bắt đầu sau khoá của target | Báo kèm `temporal`; chọn `temporal` nếu chênh lệch nhỏ |
| HSL chạy chậm | Chạy thử `ONLY="o3"` trước, ước lượng thời gian rồi mới chạy cả nhánh |
| Tinh chỉnh quá khớp validation | Chọn bằng val, test chỉ báo một lần cho cấu hình đã chốt |
