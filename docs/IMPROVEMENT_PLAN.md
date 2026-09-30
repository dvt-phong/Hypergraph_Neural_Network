# Kế hoạch nâng cấp HGNN + HSL (bản 2)

Ngày lập: 28/09/2026. Bản 2 thay cho bản 1 cùng ngày, bổ sung ba phần:
(1) chẩn đoán vì sao F1 và recall thấp, dao động mạnh; (2) đối chiếu siêu tham
số với code HGNN và HSL gốc; (3) việc cần sửa theo từng file và bảng thí nghiệm
cụ thể. Các mã H1–H11 giữ như bản 1.

Mỗi hạng mục có: **vấn đề**, **việc cần làm** (file, hàm), **thí nghiệm**, và
**tiêu chí đạt**.

---

## Cập nhật sau lần chạy 3: bước tiếp theo

### Kết quả lần 3 (HSL, cấu hình P0, `--tag p0`, 5 seed, test)

| Lần | AUC | AUPRC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| 2 (cấu hình cũ) | 0,8092 ± 0,0029 | 0,9096 ± 0,0017 | 0,4507 ± 0,3128 | 0,9226 ± 0,0127 | 0,3519 ± 0,2937 |
| **3 (P0)** | **0,8324 ± 0,0009** | **0,9236 ± 0,0003** | **0,8969 ± 0,0005** | 0,8490 ± 0,0009 | 0,9506 ± 0,0019 |
| LR, 1 seed (chạy thử local) | 0,845 | 0,925 | 0,905 | 0,862 | 0,954 |
| GBDT, 1 seed (chạy thử local) | 0,868 | 0,941 | 0,908 | 0,867 | 0,952 |

### Đọc từ `history.png` của lần 3

| Quan sát | Kết luận |
|---|---|
| BCE đi ngang ở khoảng 0,39 từ epoch 70–100; best epoch 65–115; dừng ở 165–215 | Đã train đủ. H3 đạt |
| Train AUC chỉ khoảng 0,85, chỉ hơn val 0,01–0,02 | **Underfit**, không phải overfit. Model chưa khớp nổi cả train, trong khi LR/GBDT không dùng graph đã hơn. Lan truyền trên graph đang làm mất tín hiệu của chính node |
| `kept_*` tăng từ 0,90 lên 0,98–0,99 | HSL học cách **giữ lại**, không cắt. Không có gì phạt việc giữ, nên cắt chỉ là nhiễu có hại. H\* ≈ H0 + ΔH |
| Ngưỡng tối ưu trên tập con val dao động 0,33–0,50 rồi ổn định quanh 0,45–0,50 | Giải thích vì sao ngưỡng cố định 0,5 không ổn; t\* hiện đã ổn |
| Đường val AUC lệch nhau giữa các seed nhiều hơn std test (0,0009) | Tập con 5.000 target được chọn **theo seed**, nên mỗi seed validate trên một tập khác nhau |

**Hệ quả cho cách làm thí nghiệm:** std test chỉ khoảng 0,001, nên có thể **sàng
lọc bằng 1 seed** (seed 1): chênh lệch < 0,003 AUC coi là nhiễu. Chỉ cấu hình được
chọn mới chạy lại 5 seed. Cách này tiết kiệm khoảng 5 lần thời gian GPU.

### Kế hoạch bước tiếp theo

```text
Bước A  chẩn đoán: LR, GBDT, MLP, HGNN × 5 seed        chỉ chạy, không sửa code
Bước B  encoder hết underfit: skip, trọng số family,    code ~1 ngày, sàng lọc 1 seed
        giảm regularization
        ── Cổng A/B ──
Bước C  HSL thực sự cắt: phạt độ thưa, giảm dần τ,      chỉ làm nếu HSL ≈ HGNN ở bước A
        khởi tạo giữ thấp hơn
Bước D  H7 (train trên local graph), H4 (feature),      theo Cổng A/B
        H5 (hyperedge User)
```

#### Bước A. Chẩn đoán: graph đang giúp hay đang hại? (chạy ngay)

```bash
ONLY="logreg gbdt mlp hgnn" SKIP_PREP=1 RUN_SCRIPT=scripts/run_p0.sh bash scripts/run_tmux.sh
```

| So sánh | Nếu thấy | Kết luận |
|---|---|---|
| MLP với HSL (0,8324) | MLP ≥ HSL | Lan truyền đang hại: làm B1 trước tiên |
| HGNN với HSL | \|Δ AUC\| < 0,003 | Module HSL chưa đóng góp gì (khớp với `kept` ≈ 0,99): làm bước C |
| GBDT với mọi mô hình nơ-ron | GBDT hơn rõ | Dữ liệu dạng bảng; cần B và H4 để thu hẹp khoảng cách |

**Kết quả bước A (29/09/2026, test, 5 seed):**

| Mô hình | AUC | AUPRC | F1 | Macro-F1 | F1 lớp không bỏ | Best epoch / epoch đã chạy |
|---|---|---|---|---|---|---|
| GBDT | **0,8699 ± 0,0002** | **0,9418** | 0,9089 | 0,7771 | 0,6452 | – |
| MLP | **0,8693 ± 0,0001** | 0,9410 | 0,9089 | **0,7800** | **0,6511** | 485–590 / 585–600 |
| LR | 0,8544 ± 0 | 0,9318 | 0,9065 | 0,7698 | 0,6331 | – |
| HGNN | 0,8398 ± 0,0081 | 0,9269 | 0,8991 | 0,7457 | 0,5923 | 45–600 / 145–600 |
| HSL (lần 3) | 0,8324 ± 0,0009 | 0,9236 | 0,8969 | 0,7364 | 0,5758 | 65–115 / 165–215 |

- **MLP ≥ HSL, và MLP ≈ GBDT.** Lan truyền trên graph làm mất khoảng 3 điểm AUC.
  Encoder nơ-ron không kém mô hình cây; giới hạn khoảng 0,87 đến từ feature.
- **Early stopping cắt ngang mô hình graph.** HGNN seed 111 và 11111 chạy hết 600
  epoch (best 595–600, vẫn đang tăng), đạt AUC 0,846. Seed 11 và 1111 dừng ở epoch
  145–180, chỉ đạt 0,829–0,833 (vì thế std lớn). HSL dừng ở epoch 165–215, rất có
  thể cùng hiện tượng này. MLP cần 500–600 epoch.
- **Chưa so sánh được HSL với HGNN**, vì hai bên chưa được train cùng điều kiện.
  Bước B thêm cặp `b0` / `b0-hgnn` được train dài như nhau để trả lời câu này.

#### Bước B. Sửa encoder cho khỏi underfit

| Mã | Việc | File | Chi tiết |
|---|---|---|---|
| B0 | Tập con validation giống nhau cho mọi seed | [8_train.py](../src/8_train.py) `predict` | Chọn tập con bằng `config.SPLIT_SEED` thay vì seed train, để các đường val so sánh được với nhau |
| B1 | **Skip connection** (H8, bước 2) | [5_model.py](../src/5_model.py) `HSLModel`; cờ `--skip-connection` | Thêm nhánh `h_self = MLP(X)` hai lớp, không lan truyền. `logits = Linear([Z* ‖ h_self])`. Loss contrastive vẫn dùng Z0, Z\*. Model sẽ không thể kém MLP |
| B2 | **Trọng số theo family** (H8, bước 1) | [5_model.py](../src/5_model.py) `hgnn_propagate`; cờ `--family-weights` | Mỗi family course/object/behavioral/self-loop có một trọng số học được `softplus(w_f)`, nhân vào trọng số membership. Model tự giảm ảnh hưởng của hyperedge course lớn |
| B3 | Giảm regularization | chỉ dùng cờ | Quét `--weight-decay` {5e-4, 5e-5, 0}, `--dropout` {0,5; 0,2}, `--hidden-dim` {128, 256} |

Sàng lọc với seed 1 bằng `scripts/run_b.sh`. Mọi cấu hình train tối đa 1000
epoch với patience 60 (300 epoch không cải thiện mới dừng), vì bước A cho thấy
mô hình graph hay dừng ở đoạn đi ngang tạm thời:

```bash
RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh
```

| Tên | Mô hình | Cờ | Trả lời câu hỏi |
|---|---|---|---|
| b0 | HSL | (cấu hình P0) | Khoảng cách với MLP có phải do dừng sớm? |
| b0-hgnn | HGNN | `--no-hsl` | HSL có hơn HGNN khi train như nhau? (quyết định bước C) |
| b1 | HSL | `--skip-connection` | Skip có đưa HSL lên ít nhất bằng MLP? |
| b12 | HSL | `--skip-connection --family-weights` | Trọng số family có thêm gì? |
| b12-hgnn | HGNN | `--no-hsl --skip-connection --family-weights` | HSL với HGNN, khi cả hai có skip |

Các cấu hình B3 (`b2`, `b12-wd5e-5`, `b12-wd0`, `b12-do0.2`, `b12-h256`) vẫn có sẵn
trong script, chỉ chạy khi cần: `ONLY="b12-wd0 b12-h256" RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh`.

**Mốc so sánh:** MLP seed 1, test AUC 0,8691. **Đạt khi** cấu hình tốt nhất có
test AUC ≥ 0,8691 + 0,003, tức graph đóng góp thêm vào MLP. Cấu hình đó được chạy
lại 5 seed.

#### Cổng A/B

- HSL + B ≥ GBDT: câu chuyện "graph giúp dự đoán" đã có. Làm C để HSL có đóng
  góp riêng, rồi làm D.
- HSL + B hơn MLP nhưng vẫn thua GBDT: graph có giúp, nhưng tín hiệu đầu vào còn
  yếu. Ưu tiên H4 (feature) và H5 (User).
- HSL + B vẫn ≤ MLP: vấn đề nằm ở cách dựng và dùng graph. Ưu tiên H7 (train trên
  local graph).

#### Bước C. Làm cho HSL thật sự cắt (H9, chỉ khi HSL ≈ HGNN ở bước A)

| Mã | Việc | File | Chi tiết |
|---|---|---|---|
| C1 | Phạt độ thưa | [6_hsl.py](../src/6_hsl.py), [7_losses.py](../src/7_losses.py); cờ `--beta` | Loss thêm `β · mean(p_keep)` trên membership của H0 (không tính self-loop). Quét β ∈ {0,01; 0,1} |
| C2 | Giảm dần Gumbel τ | [6_hsl.py](../src/6_hsl.py) | τ từ 1 xuống 0,1 theo epoch, thay vì cố định 0,4 |
| C3 | Khởi tạo xác suất giữ thấp hơn | [6_hsl.py](../src/6_hsl.py) `INITIAL_KEEP_LOGIT` | 3 (giữ 95%) xuống 1 (giữ 73%), để HSL thử cắt ngay từ đầu |

**Đạt khi:** `kept_*` ổn định ≤ 0,9, và HSL > HGNN (cùng bước B) ít nhất 0,003 AUC
trên 5 seed. Nếu không đạt, trình bày HSL như một ablation có kết quả âm, kèm
hình `kept_*`.

#### Bước D. Các hạng mục lớn

Làm theo Cổng A/B, chi tiết ở các mục H7, H4, H5 bên dưới.

---

## H5. Hyperedge User: kế hoạch thực hiện (01/10/2026)

**Mục tiêu.** Thêm quan hệ "cùng một học viên ở các khoá khác nhau" vào mô hình
chính (HGNN + skip + trọng số family, Course + Object; val AUC 0,8716, test
0,8737). Đây là nguồn thông tin mới duy nhất còn lại: Course và Object đã được
khai thác gần hết (mục 1 của [PHAN_TICH_HSL.md](PHAN_TICH_HSL.md)).

**Đo trước khi code** (thêm trung bình đặc trưng của các lượt đăng ký khác của cùng
học viên vào X; GBDT, val AUC):

| Đặc trưng | "any" | "temporal" |
|---|---|---|
| X | 0,8672 | 0,8672 |
| X + Course + Object | 0,8726 | 0,8726 |
| X + User | 0,8704 | 0,8690 |
| X + Course + Object + User | **0,8763** (+0,0037) | 0,8736 (+0,0010) |
| Target val có cùng học viên trong train | 75,6% | 52,4% |
| AUC của tỉ lệ bỏ học cùng học viên (target có User) | 0,672 | 0,681 |

Train có 63 277 học viên; 51,8% có ≥ 2 lượt đăng ký; trung bình 2,0, tối đa 111.

**Thiết kế** ([4_hypergraph.py](../src/4_hypergraph.py), cờ `--user-rule`):
- `any` (mặc định): mỗi học viên một hyperedge gồm mọi lượt đăng ký train của họ;
  target validation/test nối tới mọi lượt đăng ký train của cùng học viên. Dựng
  giống nhau lúc train và lúc đánh giá.
- `temporal`: mỗi lượt đăng ký một hyperedge = chính nó + các lượt đăng ký của cùng
  học viên ở khoá bắt đầu không muộn hơn. Chỉ dùng hành vi đã quan sát được tại thời
  điểm dự đoán. Dùng làm **kiểm tra độ vững**.
- Chỉ lan truyền **đặc trưng**, không bao giờ dùng nhãn của lượt đăng ký khác. Với
  `any`, đặc trưng của các khoá bắt đầu *sau* khoá của target vẫn được dùng; phải
  ghi rõ điều này khi viết, và báo kèm kết quả `temporal`.
- Hyperedge User được thêm sau Course, Object, Behavioral nên id của các hyperedge
  cũ không đổi; trọng số family có thêm `w_user`.

**Việc trên server:**

| Bước | Lệnh | Thời gian |
|---|---|---|
| U0. Dựng lại graph (dùng lại kNN, không tính lại) | `cp data/processed/simple/hypergraph.npz data/processed/simple/hypergraph_v3.npz`<br>`python src/4_hypergraph.py --reuse-neighbors hypergraph_v3.npz`<br>`python src/4_hypergraph.py --user-rule temporal --hypergraph-file hypergraph_temporal.npz --reuse-neighbors hypergraph_v3.npz` | vài phút |
| U1. Sàng lọc seed 1 | `ONLY="u0-hgnn u1-hgnn u2-hgnn u1-temporal" RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh` | khoảng 2,5 giờ |
| U2. 5 seed cho cấu hình tốt nhất theo val | `SEEDS="1 11 111 1111 11111" ONLY="<tên>" RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh` | khoảng 3 giờ |
| U3. DeLong so với mô hình chính | `python scripts/delong.py result/<U2> result/<fw2-nob-hgnn>` | vài phút |

| Tên | Cấu hình | Câu hỏi |
|---|---|---|
| u0-hgnn | HGNN + skip, Course + Object + User | User có giúp khi không có trọng số family? |
| **u1-hgnn** | u0 + trọng số family | **Mô hình chính + User** |
| u2-hgnn | HGNN + skip + trọng số family, đủ 4 loại | Model có tự tắt Behavioral khi có User? |
| u1-temporal | u1 trên graph `temporal` | Kết quả có vững khi chặt chẽ về thời gian? |

**Tiêu chí đạt:** val AUC của u1-hgnn (seed 1) ≥ 0,8740, tức hơn fw2-nob-hgnn seed 1
(0,8720) ít nhất 0,002; sau đó 5 seed hơn mô hình chính có ý nghĩa theo DeLong. Mốc
kỳ vọng từ phép đo GBDT: khoảng +0,003 đến +0,004 AUC. Nếu `temporal` chỉ tăng ít,
báo trung thực: phần lớn lợi ích đến từ các khoá cùng thời hoặc muộn hơn.

**Rủi ro:** học viên có nhiều lượt đăng ký (tối đa 111) tạo hyperedge lớn, nhưng
chỉ khoảng 0,1% học viên có trên 10 lượt. Checkpoint cũ vẫn nạp được: `8_train.py`
tự thêm mục User vào trọng số family và bộ chấm điểm HSL khi nạp.

---

## 0. Hiện trạng

Lần chạy 5 seed với cấu hình mặc định (`hidden 128`, `lr 1e-3`, `dropout 0.3`,
`pos_weight 0.319`, `λ 0.1`, `τ 0.07`, `patience 5`):

| | AUC | AUPRC | F1 | Recall |
|---|---|---|---|---|
| HSL hiện tại (test, 5 seed) | 0,809 ± 0,003 | 0,910 ± 0,002 | 0,451 ± 0,313 | 0,352 ± 0,294 |
| Đoán "ai cũng dropout" | 0,500 | 0,758 | **0,862** | 1,000 |

Mốc trong tài liệu: CFIN (Feng, Tang & Liu, AAAI 2019, Bảng 4) trên XuetangX,
cửa sổ 35 ngày. Tập dữ liệu lớn hơn của mình (467 113 so với 225 642
enrollment), nên chỉ để tham khảo.

| Mô hình | AUC (%) | F1 (%) |
|---|---|---|
| Logistic Regression | 82,23 | 89,35 |
| GBDT | 85,18 | 90,48 |
| CFIN | 86,40 | 90,92 |

---

## 1. Chẩn đoán

AUC gần như không đổi giữa các seed (± 0,003), nhưng recall dao động từ 0,03 đến
0,68. Như vậy thứ tự xếp hạng ổn định, còn **vị trí của ngưỡng 0,5** thay đổi
theo seed. Có bốn nguyên nhân, tìm được trong code.

**N1. `pos_weight = 0,319` dời ngưỡng thật lên 0,758.**
[7_losses.py:23-28](../src/7_losses.py#L23-L28) đặt `pos_weight = #âm / #dương`.
Dropout là lớp **đa số** (75,8%) nên trọng số của nó bị giảm. Với BCE có trọng
số `w`, xác suất tối ưu model học được là `p = wπ / (wπ + 1 − π)`, trong đó `π`
là xác suất thật. Thay `π = 0,758`, `w = 0,319` thì `p = 0,5`. Nghĩa là ngưỡng
0,5 trên đầu ra tương đương ngưỡng 0,758 trên xác suất thật, nên recall thấp kể
cả khi model đã hội tụ.

**N2. Model chỉ được cập nhật 10–20 lần.** Train full-batch, 1 epoch = 1 bước.
`eval_every = 5`, `patience = 5` ([8_train.py:45-46](../src/8_train.py#L45-L46))
nên dừng sau 25 bước không cải thiện; best epoch rơi vào 10–20. Model chưa train
cho xác suất khoảng 0,47 ([DATA_IO_BY_FILE.md:731](DATA_IO_BY_FILE.md)), sau
10–20 bước vẫn dồn quanh 0,5. Seed nào lệch lên một chút thì recall cao, lệch
xuống thì recall gần 0. **Đây là nguồn chính của độ lệch chuẩn 0,3.**

**N3. Chọn checkpoint theo AUC, nhưng F1 tính ở ngưỡng cố định.**
[8_train.py:210](../src/8_train.py#L210). AUC không phụ thuộc ngưỡng, nên
checkpoint được chọn có ngưỡng nằm ở đâu cũng được.

**N4. Graph lúc train khác graph lúc đánh giá.** Trong graph train, một node
nằm trong khoảng 53 hyperedge; trong local graph, target chỉ nằm trong course,
vài object, behavioral và self-loop
([TRAINING_FLOW.md mục 8](TRAINING_FLOW.md)). Chuẩn hoá `Dv^-1/2` khác nhau nên
phân phối logit trên val/test bị dịch so với train. Đây cũng là giải thích hợp
lý nhất cho việc val AUC đạt đỉnh rất sớm.

**Hệ quả phụ.** Sau khoảng 20 bước, bias của `edge_scorer` và
`membership_output` gần như vẫn là 3,0 (giữ 95%,
[6_hsl.py:36](../src/6_hsl.py#L36)). Lúc đánh giá mọi logit > 0 đều được giữ,
nên `H* ≈ H0 + ΔH`: Me và Mv hầu như chưa cắt gì. Cần kiểm tra lại bằng
`kept_*` trong `history` (Bước 0).

### Đối chiếu siêu tham số với code gốc

Ký hiệu ⚠: nhớ từ code gốc, **chưa mở lại được trong phiên này**. Em mở
`iMoonLab/HGNN/config/config.yaml` và `pkualpha/HSL` để kiểm tra trước khi đưa
vào bài.

| Tham số | Code mình | HGNN gốc ⚠ | HSL gốc (dựa trên code AllSet) | Đánh giá |
|---|---|---|---|---|
| Learning rate | 1e-3 | 1e-3 | 1e-3 ⚠ | khớp |
| Weight decay | 5e-4 | 5e-4 | 0 ⚠ | khớp HGNN |
| Hidden | 128 | 128 | tuỳ dataset | khớp |
| Dropout | 0,3 | 0,5 | 0,5 ⚠ | lệch nhẹ |
| Số epoch | ≤ 200, dừng sau 25 bước | 600, không early stop | 500, không early stop ⚠ | **lệch lớn** (N2) |
| LR schedule | không | MultiStepLR, milestone 100, γ = 0,9 | không ⚠ | lệch nhẹ |
| Loss | BCE, `pos_weight` 0,319 | CE không trọng số | CE không trọng số | **lệch lớn** (N1) |
| Quyết định lớp | ngưỡng 0,5 | argmax | argmax | **lệch lớn** (N3) |
| Chọn checkpoint | val AUC | "val" acc; trong `train.py`, phase `val` chạy trên tập test ⚠, không nên bắt chước | val acc | cần chỉ số có ngưỡng |
| Contrastive | λ 0,1, τ 0,07 | – | `contrast: False` cho cả 7 dataset (đã kiểm tra, commit `00b181d`) | cần ablation λ = 0 |
| Mask Me | có tác dụng | – | dòng `x[:…] * edge_mask` không gán lại nên Me không tác động (đã kiểm tra) | code mình đúng hơn |

**Kết luận:** các siêu tham số của mạng (lr, wd, hidden) đã khớp HGNN. Cần sửa
**giao thức train và đánh giá** (N1–N3) trước, sau đó mới tới N4 và kiến trúc.

---

## 2. Nguyên tắc làm thí nghiệm

1. **Không nhìn test khi tinh chỉnh.** Mọi lựa chọn (lr, ngưỡng, λ, …) dựa trên
   validation. Test chỉ chạy một lần cho mỗi cấu hình đã chốt.
2. **Mỗi lần chỉ đổi một thứ** so với cấu hình gốc của giai đoạn đó.
3. **Quét tham số với seed 1**, chốt cấu hình, rồi mới chạy 5 seed.
4. Mỗi cấu hình 5 seed chạy bằng `scripts/run_all.sh`, kết quả nằm ở
   `result/<ngày>/`, và được ghi vào `docs/ket_qua_thi_nghiem.xlsx` bằng
   `scripts/export_excel.py` kèm `--note` là mã thí nghiệm (E1.1, E1.2, …).
5. Mọi bảng đều báo mean ± std trên 5 seed.

---

## 3. Lộ trình

```text
Ngày 1        Bước 0   đọc lại kết quả cũ, xác nhận N1–N3
Tuần 1        P0       H1 đánh giá + ngưỡng   H3 giao thức train   H2 baseline + ablation
              ── Cổng 1 ──
Tuần 2–3      P1       H4 feature   H5 hyperedge User
              ── Cổng 2 ──
Tuần 4–5      P2       H7 train trên local graph   H6 encoder thời gian   H8 backbone   H9 HSL cắt thật
Tuần 6        P3       H10 contrastive   H11 thống kê + viết
              ── Cổng 3 ──
```

---

## Bước 0. Đọc lại kết quả cũ (không cần sửa code)

Chép thư mục `result/<ngày>/` của lần chạy 5 seed từ server về, rồi xem trong
từng `reports/*_train.json`:

| Cần xem | Nếu thấy | Kết luận |
|---|---|---|
| `best_epoch`, `len(history)` | best 10–20, dừng 35–45 | xác nhận N2 |
| `history[*].validation.recall` qua các lần validate | nhảy mạnh giữa các lần validate, dù AUC ít đổi | xác nhận N2, N3 |
| `history[*].train_auc` so với `validation.auc` | train AUC tăng tiếp, val AUC đi ngang hoặc giảm sớm | dấu hiệu N4 |
| `kept_course`, `kept_object`, `kept_behavioral` | đều khoảng 0,95 | HSL chưa cắt gì |
| `added` | khoảng `2 × số Behavioral hyperedge` | ΔH được giữ gần hết |

**Đạt khi:** có một bảng tóm tắt theo seed cho các cột trên.

---

## P0. Làm đúng trước khi làm tốt hơn (tuần 1)

### H1. Chọn ngưỡng trên validation, sửa trọng số lớp, báo đủ chỉ số

**Vấn đề:** N1, N3.

**Việc cần làm.**

| File | Thay đổi |
|---|---|
| [8_train.py](../src/8_train.py) `classification_metrics` | Nhận thêm `threshold`. Trả về: `auc`, `auprc`, `f1`, `precision`, `recall` (lớp dropout, tại ngưỡng), `f1_at_0.5`, `macro_f1`, `f1_negative`, `auprc_negative` (lớp không dropout, dùng `1 − p`). |
| [8_train.py](../src/8_train.py) hàm mới `best_threshold(labels, probabilities)` | Dùng `precision_recall_curve`, tính `F1 = 2PR / (P + R)`, trả về ngưỡng cho F1 cao nhất. |
| [8_train.py](../src/8_train.py) `evaluate` | Trả thêm mảng xác suất; lưu ra `outputs/reports/<run>_<split>_probs.npz` (nhãn + xác suất) để vẽ histogram. |
| [8_train.py](../src/8_train.py) `train` | Sau khi train xong, nạp lại best checkpoint, đánh giá trên **toàn bộ** validation, tính `t*`, ghi `t*` vào checkpoint và `_train.json`. |
| [8_train.py](../src/8_train.py) `test` | Dùng `t*` đọc từ checkpoint; vẫn báo thêm `f1_at_0.5` để so với bản cũ. |
| [7_losses.py](../src/7_losses.py) `positive_class_weight` | Thêm setting `pos_weight`: `"balanced"` (như cũ) hoặc một số (mặc định **1.0**). CLI `--pos-weight`. |
| [8_train.py](../src/8_train.py) | Setting `select_metric` ∈ {`auc`, `auprc`}, mặc định `auprc`. CLI `--select-metric`. |
| `scripts/collect_results.py`, `scripts/export_excel.py` | Thêm cột `threshold`, `macro_f1`, `f1_negative`, `auprc_negative`, `f1_at_0.5`. |
| Script mới `scripts/plot_probs.py` | Histogram xác suất val theo nhãn, 5 seed trên cùng một hình. |

**Thí nghiệm.**

| Mã | Cấu hình | Mục đích |
|---|---|---|
| E1.1 | 5 checkpoint cũ, chỉ chạy lại đánh giá với `t*` | Tách riêng ảnh hưởng của ngưỡng, không train lại |
| E1.2 | cấu hình cũ + `--pos-weight 1`, 5 seed | Ảnh hưởng của trọng số lớp |

**Đạt khi:** std của F1 test giữa các seed < 0,01, và F1 test > 0,862. Histogram
của E1.1 cho thấy 5 phân phối dồn quanh 0,5 ở các vị trí lệch nhau (hình này dùng
được trong báo cáo để giải thích kết quả bản 1).

### H3. Train đủ số bước

**Vấn đề:** N2.

**Việc cần làm.**

| File | Thay đổi |
|---|---|
| [8_train.py](../src/8_train.py) `DEFAULT_SETTINGS` | `epochs` 600, `patience` 20, `dropout` 0,5, `validation_limit` 5 000 (tập con cố định theo seed, đã có sẵn trong `evaluate`). |
| [8_train.py](../src/8_train.py) | Tuỳ chọn `--lr-schedule` ∈ {`none`, `multistep`} (milestone 100, γ 0,9, giống HGNN). |
| [8_train.py](../src/8_train.py) | Ghi thời gian mỗi epoch và mỗi lần validate vào `history` để ước lượng ngân sách chạy. |
| Script mới `scripts/plot_history.py` | Vẽ `loss`, `bce`, `contrastive`, `train_auc`, `validation.auc`, `kept_*` theo epoch. |

**Thí nghiệm** (seed 1, sau đó chạy 5 seed với cấu hình tốt nhất):

| Mã | Quét |
|---|---|
| E3.1 | lr ∈ {1e-3, 3e-3, 1e-2} |
| E3.2 | dropout ∈ {0,3; 0,5} với lr tốt nhất |
| E3.3 | `--lr-schedule multistep` với lr tốt nhất |

**Đạt khi:** đường val AUC tăng rồi đi ngang rõ ràng trước khi dừng; best epoch
> 50; std của AUC vẫn ≤ 0,005.

Từ đây gọi cấu hình tốt nhất của H1 + H3 là **cấu hình P0**. Mọi thí nghiệm sau
đều so với nó.

### H2. Baseline và ablation

**Vấn đề:** chưa biết phần cải thiện đến từ feature, từ graph hay từ HSL. Bài HSL
(Bảng 1) cho thấy MLP chỉ kém hypergraph dưới 1% trên một số dataset; trên dữ
liệu bảng, mô hình cây thường mạnh (Grinsztajn et al., NeurIPS 2022).

**Việc cần làm.**

| File | Thay đổi |
|---|---|
| [4_hypergraph.py](../src/4_hypergraph.py) `load_train_graph`, `load_evaluation_split` | Tham số `families` (mặc định tất cả): chỉ giữ membership của các family được chọn, self-loop luôn giữ. CLI `--families course,object,behavioral`. |
| MLP | Chạy `--no-hsl --families self_loop`. Khi chỉ còn self-loop, bậc node và bậc hyperedge đều bằng 1, nên HGNN hai lớp trở thành đúng MLP hai lớp, cùng code và cùng giao thức. |
| File mới `src/9_baselines.py` | Logistic Regression và GBDT (`HistGradientBoostingClassifier` của scikit-learn, cùng họ histogram GBDT với LightGBM, không cần cài thêm) trên `X.npy`, chọn `t*` trên validation, báo cùng bộ chỉ số như H1. |

**Thí nghiệm** (5 seed, cấu hình P0):

| Mã | Nhóm | Cấu hình |
|---|---|---|
| E2.1 | Không graph | LR, LightGBM |
| E2.2 | Không graph | MLP (`--no-hsl --families self_loop`) |
| E2.3 | Graph, không học cấu trúc | HGNN (`--no-hsl`) |
| E2.4 | Ablation HSL | `--no-edge-sampling` |
| E2.5 | Ablation HSL | `--no-node-sampling` |
| E2.6 | Ablation HSL | `--add-per-edge 0` |
| E2.7 | Ablation HSL | `--lambda-cl 0` |
| E2.8 | Ablation family | bỏ lần lượt Course / Object / Behavioral |
| E2.9 | Ablation feature | `--feature-set behavior / behavior_user / behavior_course` |

**Đạt khi:** có bảng trả lời được ba câu hỏi: graph có hơn LightGBM không, HSL có
hơn HGNN không, và thành phần nào đóng góp nhiều nhất.

### Cổng 1 (cuối tuần 1)

- LightGBM ≥ HGNN/HSL: làm P1 (feature, User hyperedge) trước, chưa đụng kiến
  trúc.
- HGNN > LightGBM nhưng HSL ≈ HGNN: vẫn làm P1, và đưa H9 lên sớm.
- HSL > HGNN > LightGBM: làm P1 rồi P2 theo thứ tự.

---

## P1. Tăng tín hiệu đầu vào (tuần 2–3)

### H4. Bổ sung feature theo CFIN

**Vấn đề.** `X` hiện chỉ có số event theo ngày, số lần theo action, và
user/course dạng one-hot. `session_id` có trong log nhưng bị bỏ ở bước 2; chưa có
tỉ lệ làm bài đúng, độ gần đây, số ngày hoạt động. CFIN (Bảng 3, 5) cho thấy thời
lượng và số session, tỉ lệ trả lời đúng, và cách tổng hợp video ảnh hưởng rõ tới
dropout.

**Việc cần làm.**

| File | Thay đổi |
|---|---|
| [2_preprocess.py](../src/2_preprocess.py) `stream_events` | Giữ `session_id` trong CSV của từng split. |
| [3_features.py](../src/3_features.py) `build_split_features` | Thêm các khối feature ở bảng dưới, đặt sau khối behavior. |
| [0_config.py](../src/0_config.py) | Cập nhật số cột và vị trí các khối; `feature_columns` thêm tập `full_v2`. |
| [4_hypergraph.py](../src/4_hypergraph.py) | kNN của Behavioral vẫn chỉ dùng khối behavior cũ, để H0 không đổi khi so sánh. |

| Feature | Cách tính |
|---|---|
| Session | số session, tổng và trung bình thời lượng session |
| Hoạt động | số ngày có hoạt động, ngày hoạt động cuối (recency), chuỗi ngày liên tiếp dài nhất, số event theo tuần |
| Assignment | `problem_check_correct / problem_check`, số lần reset trên mỗi bài |
| Video | số video khác nhau đã xem / tổng số video của khoá |
| Context theo khoá | (feature − trung bình của khoá) / std của khoá, **thống kê chỉ tính trên train** |
| Context theo user | trung bình / max của cùng user trên các enrollment khác (chỉ feature, không nhãn) |

**Thí nghiệm.** E4.1: LightGBM, MLP, HGNN, HSL trên `full_v2`, 5 seed.

**Đạt khi:** LightGBM trên `full_v2` đạt AUC ≥ 0,83; HGNN và HSL tăng tương ứng.

### H5. Hyperedge User (cùng một người, khác khoá)

**Vấn đề.** Đo trên dữ liệu của mình: khoảng 52% user trong train có ≥ 2
enrollment; 75,6% enrollment validation và 76,1% enrollment test có cùng user
trong train. CFIN (trang 519–520) báo xác suất dropout của một user giữa các khoá
tương quan dương rõ rệt. Graph hiện tại chưa nối các enrollment này.

**Việc cần làm.**

| File | Thay đổi |
|---|---|
| [0_config.py](../src/0_config.py) | `EDGE_FAMILIES = ("course", "object", "behavioral", "user", "self_loop")`. |
| [4_hypergraph.py](../src/4_hypergraph.py) `build_train_hyperedges` | Mỗi user có ≥ 2 enrollment train tạo một hyperedge User. |
| [4_hypergraph.py](../src/4_hypergraph.py) `load_evaluation_split`, `build_local_graph` | `members_by_key` thêm khoá user; target tham gia hyperedge User gồm các enrollment train của cùng user. |
| [6_hsl.py](../src/6_hsl.py) | Không cần sửa: `edge_scorer` đã nhận one-hot family nên tự thêm được family mới. ΔH vẫn chỉ áp cho Behavioral. |

**Rủi ro.** Chỉ lan truyền feature, **không** dùng nhãn của enrollment khác làm
feature, vì trong thực tế các khoá có thể diễn ra cùng lúc. Nếu cần chặt chẽ về
thời gian, chỉ nối với các khoá bắt đầu trước hoặc cùng lúc với khoá của target.

**Thí nghiệm.** E5.1: HGNN và HSL, có và không có User, 5 seed. E5.2: User giới
hạn trong cùng category.

**Đạt khi:** AUC tăng nhiều hơn std giữa các seed. Nếu đạt, đây là đóng góp riêng
về cấu trúc có thể viết thành bài.

### Cổng 2 (cuối tuần 3)

- Hypergraph (nhất là có User) hơn LightGBM trên cùng bộ feature: câu chuyện của
  luận án là *quan hệ bậc cao giữa các enrollment giúp dự đoán dropout*. Làm P2.
- Hypergraph không hơn LightGBM: ưu tiên H7 (sửa N4) trước khi kết luận, vì graph
  đang bị đánh giá trên phân phối khác lúc train.

---

## P2. Mô hình và cách train (tuần 4–5)

### H7. Train trên local graph, giống lúc đánh giá (ưu tiên cao nhất của P2)

**Vấn đề:** N4, và đồng thời N2 (quá ít bước cập nhật).

**Việc cần làm.** Hai phương án:

- **(A) Inductive, giữ thiết lập hiện tại.** Mỗi bước lấy mẫu 256 node train làm
  target. Dựng local graph cho từng target bằng đúng `build_local_graph`, nhưng
  **loại chính target khỏi các hyperedge của nó** (target đóng vai "học viên
  mới", giống val/test). Mỗi epoch có khoảng 490 bước.
  - [4_hypergraph.py](../src/4_hypergraph.py): hàm `load_train_split_as_targets`
    trả về cùng cấu trúc như `load_evaluation_split` cho split train, kèm danh
    sách láng giềng đã loại chính node.
  - [8_train.py](../src/8_train.py): vòng lặp mini-batch dùng
    `merge_local_graphs`; contrastive loss lấy anchor trong batch.
- **(B) Transductive.** Đưa node val/test vào graph như node **không có nhãn**.
  Không lộ nhãn, và đúng với thực tế là khi dự đoán thì log 35 ngày của cả khoá
  đã có. Bài HSL dùng cách này. Đây là thay đổi về thiết lập thí nghiệm nên cần
  thống nhất với thầy hướng dẫn và ghi rõ khi viết.

Làm (A) trước vì giữ được thiết lập hiện tại.

**Bằng chứng.** GraphSAGE (NeurIPS 2017) và ShaDow-GNN (NeurIPS 2021) dùng cùng
một bộ trích subgraph cục bộ cho cả train và suy luận; GraphSAINT (ICLR 2020) và
Cluster-GCN (KDD 2019) train mini-batch trên subgraph; SIG-Net cũng dựng subgraph
cho từng cặp learner–course.

**Thí nghiệm.** E7.1: HGNN và HSL theo (A), 5 seed, so với cấu hình P0.

**Đạt khi:** khoảng cách AUC train/val hẹp lại; AUC val tăng; `t*` gần 0,5 hơn
(logit không còn bị dịch).

### H6. Encoder theo thời gian cho 35 ngày

**Vấn đề.** 35 cột theo ngày đi qua một `Linear` như các cột độc lập, nên model
không biết ngày 5 đứng sau ngày 4. Các bài dropout đều có module thời gian (Fei &
Yeung 2015; CFIN; MST-GCN; CA-TFHN).

**Việc cần làm.** Bước 3 lưu thêm tensor `(N, 35, số action)`. Trong
[5_model.py](../src/5_model.py), thay `layer1` bằng: 1D-CNN hoặc GRU trên chuỗi
ngày ‖ MLP cho phần tĩnh, ghép lại rồi mới vào HGNN.

**Đạt khi:** AUC tăng ở cả MLP và HGNN so với H4.

### H8. Backbone: trọng số theo family và chống over-smoothing

**Vấn đề.** Course hyperedge có 151–2 361 thành viên, lấy trung bình đều; ba
family đang có trọng số như nhau (`W = I`, trong khi HGNN có ma trận `W`).

**Việc cần làm** (theo thứ tự tăng độ phức tạp, dừng khi hết cải thiện):
1. Học trọng số `w_f` cho từng family trong `hgnn_propagate`
   ([5_model.py:61](../src/5_model.py#L61)).
2. Thêm residual `Z = HGNN(X) + XΘ` (kiểu UniGCNII).
3. Thay backbone bằng AllSetTransformer (backbone gốc của HSL).

**Đạt khi:** AUC tăng; độ over-smoothing giảm (cosine trung bình giữa các thành
viên cùng khoá thấp hơn).

### H9. Làm cho HSL thật sự học cấu trúc

**Vấn đề.** Bias khởi tạo cho xác suất giữ khoảng 0,95 và không có gì thúc HSL cắt
bớt. Bài HSL báo tỉ lệ pruning `r^v` từ 11% đến 88%; PTDNet (WSDM 2021) cần phạt
độ thưa thì mới loại được cạnh nhiễu.

**Việc cần làm** trong [6_hsl.py](../src/6_hsl.py) và
[7_losses.py](../src/7_losses.py):
1. Log `r^v` (tỉ lệ membership bị cắt) theo family.
2. Thêm phạt `β · mean(p_keep)` hoặc ngân sách giữ lại theo family; CLI `--beta`.
3. Giảm dần Gumbel τ từ 1 xuống 0,1 (Jang et al., ICLR 2017), thay vì cố định 0,4.
4. Phân tích định tính: family nào hay bị cắt (object hiếm? course lớn?).

**Đạt khi:** `r^v` rõ ràng lớn hơn 0 và AUC không giảm. Có một hình phân tích
cấu trúc bị cắt để đưa vào bài.

---

## P3. Hoàn thiện (tuần 6)

### H10. Contrastive: augmentation và nhiệt độ

**Vấn đề.** Hai view Z0 và Z\* gần như trùng nhau khi `H* ≈ H0`; τ = 0,07 rất
thấp so với chỉ 64 cặp âm. Bản HSL công bố tắt contrastive.

**Việc cần làm.** Thêm mask feature và mask theo ngày cho một view; quét
τ ∈ {0,07; 0,2; 0,5} và λ ∈ {0; 0,05; 0,1; 0,5}.

**Đạt khi:** có ít nhất một cấu hình hơn λ = 0 có ý nghĩa thống kê. Nếu không có,
bỏ contrastive và ghi rõ trong bài.

### H11. Thống kê và cách báo cáo

- Kiểm định DeLong cho AUC giữa mô hình chính và baseline tốt nhất; Wilcoxon theo
  seed cho F1 và AUPRC.
- Báo thời gian train, bộ nhớ GPU; ECE nếu có hiệu chỉnh xác suất.
- Bảng chính: LR, LightGBM, MLP, HGNN, HSL, HSL + User; các cột AUC, AUPRC,
  F1 (tại `t*`), macro-F1, F1 lớp không dropout.

### Cổng 3 (cuối tuần 6)

Chốt cấu hình, chạy lại toàn bộ bảng thí nghiệm bằng `scripts/run_tmux.sh`, rồi
viết. Nếu HSL không hơn HGNN, trình bày HSL như một ablation có kết quả âm, kèm
phân tích `r^v` và `kept_*`.

---

## 4. Bảng theo dõi

Code cho H1, H3 và H2 (trừ ablation feature mới) đã có từ 28/09/2026; các thí
nghiệm cần chạy trên server.

| Mã | Hạng mục | Trạng thái | Kết quả chính |
|---|---|---|---|
| B0 | Đọc lại kết quả cũ | chưa làm | |
| H1 | Code: t\*, `--pos-weight`, `--select-metric`, chỉ số mới, `plot_results.py` | xong | |
| H3 | Code: `patience` 20, `dropout` 0,5, `--lr-schedule`, thời gian mỗi epoch | xong | |
| H2 | Code: `--families`, `9_baselines.py` | xong | |
| E1.1 | Đánh giá lại với `t*` | chưa chạy | `8_train.py --mode test --checkpoint <cũ>.pt` |
| E1.2 + H3 | HSL, cấu hình P0 (lần 3) | xong, 5 seed | AUC 0,8324 ± 0,0009; F1 0,8969 ± 0,0005. Train AUC ≈ 0,85 (underfit); `kept_*` ≈ 0,99 |
| A | LR, GBDT, MLP, HGNN × 5 seed | xong | AUC: GBDT 0,8699; MLP 0,8693; LR 0,8544; HGNN 0,8398 ± 0,0081. Graph đang làm giảm AUC |
| B0–B2 | Code: tập con val cố định, `--skip-connection`, `--family-weights` | xong | |
| B (run_b.sh) | b0, b0-hgnn, b1, b12, b12-hgnn; seed 1; 1000 epoch, patience 60 | xong | AUC: b0 0,8332; b0-hgnn 0,8467; **b1 0,8699**; b12 0,8697; b12-hgnn 0,8665. Skip +3,7 điểm nhưng mới ngang MLP (0,8691). Phân tích: [PHAN_TICH_HSL.md](PHAN_TICH_HSL.md) |
| B-nob | b1-nob, b1-nob-hgnn (bỏ Behavioral) | xong, seed 1 | **b1-nob-hgnn 0,8730** (> MLP 0,8691 và GBDT 0,8699); b1-nob 0,8692 |
| C0 | b1-nob-hgnn × 5 seed | xong | **AUC 0,8729 ± 0,0001; AUPRC 0,9449 ± 0,0001**; F1 0,9098; macro-F1 0,7765. Vượt MLP +0,0036 và GBDT +0,0030 AUC |
| C0-ab | Ablation b1-hgnn; trọng số family bản sửa fw2-hgnn, fw2-nob-hgnn (seed 1) | xong | b1-hgnn 0,8665; **fw2-hgnn 0,8728**; **fw2-nob-hgnn 0,8740** (tốt nhất). Trọng số family bản sửa tự khắc phục Behavioral |
| C1 | fw2-hgnn, fw2-nob-hgnn × 5 seed | fw2-hgnn xong; fw2-nob-hgnn chờ kết quả | fw2-hgnn: AUC 0,8723 ± 0,0012 (seed 111: 0,8702, dừng sớm); val 0,8705 ± 0,0006 |
| C2 | Chốt mô hình chính bằng val AUC 5 seed; chạy lại 1500 epoch; kiểm định DeLong | chốt xong, DeLong xong; còn chạy 1500 epoch | **Mô hình chính: fw2-nob-hgnn**, val 0,8716 ± 0,0004, test AUC 0,8737 ± 0,0002. DeLong hơn GBDT p ≤ 6·10⁻⁷, hơn MLP p ≤ 2·10⁻¹² (5/5 seed) |
| C3 | Hyperedge User (H5), rồi trọng số `α_e` cho từng hyperedge | chưa làm | |
| C1–C3 | HSL cắt thật | chờ bước A | |
| E3.1–E3.3 | lr, dropout, schedule | chưa chạy | |
| E2.1 | LR, GBDT (chạy thử 1 seed trên máy local, 28/09) | xong 1 seed | Test: LR AUC 0,845, F1 0,905; GBDT AUC 0,868, AUPRC 0,941, F1 0,908. **Cả hai đều cao hơn HSL hiện tại (AUC 0,809).** Cần chạy đủ 5 seed |
| E2.2–E2.9 | MLP, HGNN, ablation | chưa chạy | |
| E4.1 | Feature `full_v2` | chưa làm | |
| E5.1–E5.2 | Hyperedge User | chưa làm | |
| E7.1 | Train trên local graph | chưa làm | |
| H6, H8, H9, H10 | Mô hình | chưa làm | |

---

## 5. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Validate trên 31 589 local graph tốn thời gian | `validation_limit 5000` trong lúc train; chỉ đánh giá toàn bộ val một lần cho best checkpoint |
| Lộ nhãn qua feature context hoặc User hyperedge | Thống kê chỉ tính trên train, không bao giờ dùng nhãn làm feature |
| Tinh chỉnh quá khớp validation | Quét với 1 seed, báo 5 seed; test chỉ chạy một lần cho cấu hình đã chốt |
| Số ⚠ trong bảng đối chiếu sai | Mở lại `config.yaml` của HGNN và code HSL trước khi đưa vào bài |

---

## Tài liệu dẫn

Đã đối chiếu trực tiếp nội dung:

- Feng, W., Tang, J., Liu, T. X. *Understanding Dropouts in MOOCs.* AAAI 2019,
  517–524. https://ojs.aaai.org/index.php/AAAI/article/view/3825 (Bảng 3, 4, 5).
- Cai, D. et al. *Hypergraph Structure Learning for Hypergraph Neural Networks.*
  IJCAI 2022, 1923–1929. https://doi.org/10.24963/ijcai.2022/267 (Bảng 1).
  Code: https://github.com/pkualpha/HSL (commit `00b181d`).
- Feng, Y. et al. *Hypergraph Neural Networks.* AAAI 2019, 3558–3565.
  https://doi.org/10.1609/aaai.v33i01.33013558. Code:
  https://github.com/iMoonLab/HGNN.

Dẫn theo hiểu biết chung, cần mở bản gốc để kiểm tra trước khi đưa vào bài:

- Lipton, Elkan, Naryanaswamy. *Optimal Thresholding of Classifiers to Maximize
  F1 Measure.* ECML-PKDD 2014.
- Saito, Rehmsmeier. *The Precision-Recall Plot Is More Informative than the ROC
  Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE 2015.
- Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.
- Grinsztajn, Oyallon, Varoquaux. *Why do tree-based models still outperform deep
  learning on typical tabular data?* NeurIPS 2022 (Datasets & Benchmarks).
- Shchur et al. *Pitfalls of Graph Neural Network Evaluation.* NeurIPS 2018 R2L
  Workshop.
- Fei, Yeung. *Temporal Models for Predicting Student Dropout in MOOCs.* ICDMW
  2015.
- Hamilton, Ying, Leskovec. *Inductive Representation Learning on Large Graphs.*
  NeurIPS 2017.
- Zeng et al. *Decoupling the Depth and Scope of Graph Neural Networks.* NeurIPS
  2021.
- Zeng et al. *GraphSAINT.* ICLR 2020.
- Chiang et al. *Cluster-GCN.* KDD 2019.
- Gao et al. *HGNN+: General Hypergraph Neural Networks.* IEEE TPAMI 2023.
- Huang, Yang. *UniGNN.* IJCAI 2021.
- Chien et al. *You are AllSet.* ICLR 2022.
- Luo et al. *Learning to Drop: Robust Graph Neural Network via Topological
  Denoising.* WSDM 2021.
- Jang, Gu, Poole. *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017.
- You et al. *Graph Contrastive Learning with Augmentations.* NeurIPS 2020.
- Wang, Liu. *Understanding the Behaviour of Contrastive Loss.* CVPR 2021.
- DeLong, DeLong, Clarke-Pearson. *Comparing the Areas under Two or More
  Correlated ROC Curves.* Biometrics 1988.
