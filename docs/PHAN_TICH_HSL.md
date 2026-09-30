# Vì sao HGNN + HSL chưa tốt: phân tích nguyên nhân

Ngày: 29/09/2026. Tài liệu này giải thích vì sao mô hình graph thua MLP dùng cùng
đặc trưng, dựa trên kết quả bước A và các phép đo trực tiếp trên `hypergraph.npz`.
Script đo nằm ngoài repo; các con số có thể đo lại bằng mô tả ở từng mục.

## 0. Hiện tượng cần giải thích

Test, 5 seed (bước A và lần 3):

| Mô hình | AUC | Ghi chú |
|---|---|---|
| GBDT | 0,8699 ± 0,0002 | không graph |
| MLP | 0,8693 ± 0,0001 | cùng encoder, chỉ có self-loop |
| HGNN | 0,8398 ± 0,0081 | thêm Course, Object, Behavioral |
| HSL | 0,8324 ± 0,0009 | thêm Me, Mv, ΔH, contrastive |

Thêm graph làm **mất khoảng 3 điểm AUC**; thêm HSL mất thêm gần 1 điểm. Câu hỏi
là: graph không có thông tin, hay graph có thông tin nhưng bị dùng sai?

---

## 1. Hyperedge có mang thông tin ngoài X không?

**Cách đo.** Với mỗi node, lấy trung bình đặc trưng của các thành viên khác trong
hyperedge của nó theo từng family (node train: bỏ chính nó ra; target validation:
chỉ dùng thành viên train, đúng như local graph). Ghép vào `X` rồi train LR và
GBDT, đo AUC trên validation. Đây là cách dùng graph đơn giản nhất, không lan
truyền nhiều lớp.

| Đặc trưng | LR | GBDT |
|---|---|---|
| X | 0,8457 | 0,8672 |
| X + trung bình Course | 0,8556 | 0,8725 |
| X + trung bình Object | **0,8588** | 0,8717 |
| X + trung bình Behavioral | 0,8534 | **0,8666** |
| X + cả ba | **0,8637** | **0,8728** |

(LR ở đây có chuẩn hoá lại toàn bộ cột nên số của dòng X khác baseline LR ở bước
A; chỉ so sánh trong bảng này.)

**Độ đồng nhãn** (AUC khi dùng tỉ lệ bỏ học của các thành viên khác làm điểm dự
đoán, validation): Course 0,731; Object 0,720; **Behavioral 0,824**.

**Kết luận.**
- **Course và Object có thông tin thật** mà X không có: GBDT tăng khoảng +0,5 điểm,
  LR tăng +1,0 đến +1,3 điểm. X chỉ có category của khoá, không có mức độ hoạt
  động chung của khoá hay của nhóm cùng xem một video.
- **Behavioral không thêm gì** (GBDT 0,8666 so với 0,8672), dù độ đồng nhãn cao
  nhất. Lý do: hàng xóm Behavioral được chọn bằng cosine trên chính các cột hành vi
  của X, nên "hàng xóm giống mình" chỉ là một bộ phân loại kNN trên X, thông tin mà
  MLP và GBDT đã có sẵn.
- Vậy **graph có thông tin, nhưng bị dùng sai**. Mức trần của cách dùng đơn giản
  nhất đã là khoảng MLP + 0,5 điểm, trong khi HGNN lại thấp hơn MLP 3 điểm.

---

## 2. Nguyên nhân 1: đặc trưng của chính node bị pha loãng

**Cách đo.** Với phép lan truyền `Dv^-1/2 H De^-1 Hᵀ Dv^-1/2`, mỗi hyperedge của
node v nhận khoảng `1/d_v` khối lượng, và trong hyperedge e phần của chính v là
`1/|e|`. Cộng lại ra phần đặc trưng của chính node trong đầu ra (xấp xỉ).

| | Trung vị | p10 |
|---|---|---|
| Bậc `d_v` của node train (kể cả self-loop) | 18 | 5 |
| Phần của chính node sau **1** lớp, train | **12%** | 4,8% |
| Phần của chính node sau **2** lớp, train (xấp xỉ) | **khoảng 1,5%** | |
| Phần của chính node sau 1 lớp, target validation | 16,7% | 2,5% |

Dropout chủ yếu là **hành vi cá nhân** (số ngày học, số lần xem video...). Sau hai
lớp HGNN, gần như toàn bộ biểu diễn của một học viên là trung bình của người khác,
và chỉ khoảng 1,5% là của chính họ. MLP thì giữ 100%. Đây là lý do chính khiến HGNN
thua MLP, và cũng là lý do train AUC chỉ đạt khoảng 0,85 (underfit): model không
còn nhìn thấy đủ rõ từng học viên.

Hyperedge còn rất lớn: Course trung vị 377 thành viên (tối đa 2 361); Object trung
vị 27 nhưng p99 là 757. Lấy trung bình trên vài trăm người thì mọi thành viên nhận
cùng một thông điệp (over-smoothing, Li, Han & Wu, AAAI 2018).

**Cách sửa:** skip connection (B1, đã code), hoặc attention cho phép node tự chọn
trọng số của self-loop (mục 7).

---

## 3. Nguyên nhân 2: hyperedge Behavioral vừa thừa vừa chiếm phần lớn thông điệp

Tỉ lệ đóng góp của từng family trong một lần lan truyền, node train (trung bình):

| Family | Phần thông điệp | Thông tin thêm ngoài X (mục 1) |
|---|---|---|
| Behavioral | **51%** | **không có** |
| Object | 33% | có |
| Course | 8% | có |

Vì sao Behavioral chiếm nhiều như vậy: mỗi node train có hyperedge Behavioral của
chính nó, **cộng thêm** hyperedge của mọi node khác chọn nó làm hàng xóm. Trung
bình một node nằm trong 11 hyperedge Behavioral. Phân phối rất lệch: p99 là 41,
tối đa **1 879**. Đây là hiện tượng *hubness* của kNN trong không gian nhiều chiều:
một số học viên (thường là nhóm gần như không hoạt động, vector hành vi gần giống
nhau) là "hàng xóm" của hàng nghìn người, nên đặc trưng của họ được trộn vào rất
nhiều node.

Như vậy family có ích nhất (Course, 8%) lại nhận ít trọng số nhất, còn family không
thêm thông tin (Behavioral, 51%) lại nhận nhiều nhất. ΔH còn thêm 2 node vào **mỗi**
hyperedge Behavioral (khoảng 229 000 membership), càng làm phần thừa này nặng
thêm.

**Cách sửa:** trọng số theo family (B2, đã code) để model tự giảm Behavioral; chạy
ablation bỏ hẳn Behavioral (`--families course,object`); nếu giữ lại thì dựng lại
bằng mutual kNN hoặc giới hạn số hyperedge Behavioral mỗi node được tham gia.

---

## 4. Nguyên nhân 3: graph lúc train khác graph lúc đánh giá

| | Node train | Target validation |
|---|---|---|
| Bậc `d_v`, trung vị | **18** | **6** |
| Bậc `d_v`, trung bình | 25,3 | 15,3 |
| Phần của Behavioral | 51% | khoảng 17% (chỉ 1 hyperedge của chính nó) |
| Phần của Object | 33% | **52%** |

Lúc train, biểu diễn của một node chủ yếu đến từ Behavioral. Lúc đánh giá, target
không nằm trong hyperedge Behavioral của ai (nó không thuộc tập train), nên biểu
diễn chủ yếu đến từ Object. Lớp phân loại được học trên một kiểu biểu diễn nhưng
lại được áp dụng cho một kiểu khác. Đây là lý do ngưỡng t\* của HGNN (0,32–0,39)
lệch xa 0,5, và val AUC đi ngang rất sớm.

**Cách sửa:** train trên local graph dựng giống hệt lúc đánh giá (H7). Nếu bỏ
Behavioral thì phần lớn độ lệch này cũng mất theo.

---

## 5. Nguyên nhân 4: HSL không học được cách cắt

Log lần 3 cho thấy `kept_*` tăng từ 0,90 lên 0,98–0,99, tức HSL học cách **giữ
lại**. Có bốn lý do cộng dồn:

**(a) Không có gì thưởng cho việc cắt.** Khi train, mỗi lần Gumbel cắt ngẫu nhiên
là một nhiễu làm BCE tăng. Cách giảm loss dễ nhất là giữ tất cả. Bài HSL gốc có
cấu hình khác (transductive, mỗi hyperedge một tham số, backbone AllSet) và trong
code công bố thì mask Me không có tác dụng (xem [references.md](references.md)),
nên không có bằng chứng rằng cơ chế này tự cắt được nếu không có phạt độ thưa.

**(b) Gradient bị bão hoà.** Gradient straight-through là `σ'((logit + nhiễu)/τ)/τ`
với τ = 0,4. Đo bằng mô phỏng 2 triệu mẫu:

| Xác suất giữ | Gradient trung bình | Tỉ lệ mẫu có gradient > 0,1 |
|---|---|---|
| 50% | 0,225 | 56% |
| 95% (khởi tạo) | 0,053 | 13% |
| 99% (cuối lần 3) | **0,013** | **3%** |

Khởi tạo ở mức giữ 95% đã làm gradient nhỏ đi khoảng 4 lần. Khi xác suất giữ trôi
lên 99%, gradient nhỏ đi khoảng 18 lần. Đây là vòng luẩn quẩn: giữ càng nhiều thì
càng khó học cách cắt.

**(c) Một membership ảnh hưởng quá ít.** Trong hyperedge Course 377 người, bỏ một
người chỉ đổi trung bình khoảng 0,3%. Gradient cho Mv tỉ lệ với `1/|e|`, gần như
bằng 0 trên các hyperedge lớn, tức đúng những chỗ cần cắt nhất.

**(d) Đầu vào của bộ chấm điểm đã bị làm mượt.** Me và Mv chấm điểm dựa trên `Z0`
và `h_e = mean(Z0)`, mà `Z0` chính là biểu diễn đã bị pha loãng ở mục 2. Các
hyperedge trông giống nhau nên rất khó phân biệt cái tốt với cái xấu.

Ngoài ra, lúc đánh giá HSL giữ mọi phần tử có xác suất > 0,5, tức giữ gần như tất
cả. Kết quả: H\* ≈ H0 + ΔH, và ΔH lại thêm vào đúng family thừa (mục 3).

---

## 6. Các nguyên nhân phụ

- **Contrastive gần như không có tác dụng.** Khi H\* ≈ H0 thì hai view Z0 và Z\*
  gần trùng nhau, nên cặp dương trở nên tầm thường. Cặp âm lại là các thành viên
  cùng hyperedge, tức đẩy xa đúng những node mà phép lan truyền đang kéo lại gần;
  hai lực này ngược nhau. Bản HSL công bố đặt `contrast: False`.
- **Early stopping dừng sớm** (bước A): HGNN seed nào chạy hết 600 epoch đạt
  0,846, seed nào dừng ở khoảng epoch 150 chỉ đạt 0,83. Bước B đã tăng lên 1000
  epoch với patience 60.

---

## 7. Hướng đề xuất: trọng số hyperedge học được và giải thích được

Ý tưởng "một ma trận cho biết hyperedge nào tốt, hyperedge nào xấu" chính là ma
trận `W` trong công thức HGNN, `Dv^-1/2 H W De^-1 Hᵀ Dv^-1/2`. Code hiện đang cố
định `W = I`. Đề xuất học `W` ở ba mức, mức sau chi tiết hơn mức trước:

| Mức | Trọng số | Số tham số / đầu ra | Giải thích được gì | Trạng thái |
|---|---|---|---|---|
| 1 | Theo family: `w_f` | 4 số | Loại quan hệ nào có ích | **Đã code** (`--family-weights`) |
| 2 | Theo từng hyperedge: `α_e = σ(g(e))` | 148 348 số | Khoá nào, video nào là hyperedge tốt hoặc xấu | Đề xuất |
| 3 | Theo từng cặp node–hyperedge: `β_{v,e} = softmax_{e ∋ v}(a(z_v, h_e))` | mỗi học viên một phân phối | Với học viên này, quan hệ nào quan trọng nhất, kể cả chính họ (self-loop) | Đề xuất |

**Mức 2: "HSL mềm" cho hyperedge.** Thay mask Gumbel 0/1 của Me bằng một trọng số
liên tục `α_e ∈ (0, 1)` đưa thẳng vào `W`. Như vậy gradient không bị bão hoà (mục
5b), và `α_e` đọc được trực tiếp như một điểm "tốt/xấu". Đầu vào của `g` nên gồm
one-hot family, `log |e|`, và trung bình **đặc trưng gốc X** của thành viên (không
dùng Z0 đã bị làm mượt). Không dùng nhãn của thành viên làm đầu vào, để tránh lộ
nhãn. Có thể thêm phạt entropy để `α_e` dứt khoát hơn, gần 0 hoặc gần 1.

**Mức 3: attention theo node** (kiểu HyperGAT, Ding et al., EMNLP 2020; AllSet,
Chien et al., ICLR 2022). Mỗi node tự phân phối trọng số trên các hyperedge của
mình, **kể cả self-loop**. Mức này giải quyết cùng lúc mục 2 (node tự giữ phần của
mình khi cần) và mục 3 (tự giảm Behavioral). Nó cũng thay thế tự nhiên cho Mv: Mv
hiện đã chấm điểm từng cặp `(v, e)`, chỉ cần chuyển từ mask cứng sang softmax.

**Cách kiểm tra "tốt/xấu" có thật hay không** (phân tích cho luận án):
1. Xuất bảng `α_e` sau khi train, kèm family, khoá hoặc video, kích thước, tỉ lệ bỏ
   học của thành viên. Nhãn chỉ dùng ở bước phân tích, không đưa vào model.
2. Xem `α_e` có tương quan với độ "thuần" của hyperedge (tỉ lệ bỏ học gần 0 hoặc
   gần 1) và nghịch biến với kích thước hay không.
3. **Kiểm tra độ trung thực:** bỏ 10% hyperedge có `α_e` cao nhất, rồi bỏ 10% thấp
   nhất, và so sánh AUC giảm bao nhiêu. Nếu bỏ nhóm "tốt" làm AUC giảm nhiều hơn rõ
   rệt, thì trọng số thực sự phản ánh độ hữu ích.
4. Với mức 3: trung bình `β` theo family, và theo nhóm học viên (bỏ học hay không,
   hoạt động nhiều hay ít).

Phần này có thể trở thành đóng góp chính của chương: *học trọng số quan hệ bậc cao
một cách giải thích được cho bài toán dự đoán bỏ học*. Kèm theo là một phân tích
âm có số liệu: vì sao mask cứng kiểu HSL không cắt được trên dữ liệu này (mục 5).

---

## 8. Thứ tự việc tiếp theo

| # | Việc | Code | Kiểm tra nguyên nhân |
|---|---|---|---|
| 1 | Chờ kết quả `run_b.sh` (b0, b0-hgnn, b1, b12, b12-hgnn) | có sẵn | 1, 2, 6 |
| 2 | Ablation bỏ Behavioral: HGNN và HSL với `--skip-connection --families course,object` | có sẵn | 3, 4 |
| 3 | Mức 2: trọng số mềm theo hyperedge, xuất bảng `α_e` | khoảng 1 ngày | 5, 7 |
| 4 | Mức 3: attention theo node, kể cả self-loop | khoảng 1–2 ngày | 2, 3, 5 |
| 5 | Train trên local graph (H7) | khoảng 2 ngày | 4 |
| 6 | Hyperedge User (H5) | khoảng 1–2 ngày | thêm thông tin mới |

Mục tiêu định lượng: vượt MLP (0,8693) ít nhất 0,003 AUC. Mục 1 cho thấy mức này
khả thi, vì chỉ cần dùng trung bình Course và Object một cách đơn giản, GBDT đã tăng
khoảng 0,5 điểm.

## 9. Kiểm chứng bằng bước B (seed 1, test, 1000 epoch, patience 60)

| Cấu hình | AUC | So với MLP seed 1 (0,8691) | Best / đã chạy |
|---|---|---|---|
| b0: HSL | 0,8332 | −3,6 điểm | 90 / 390 |
| b0-hgnn: HGNN | 0,8467 | −2,2 điểm | 840 / 1000 |
| b1: HSL + skip | **0,8699** | +0,08 | 355 / 655 |
| b12: HSL + skip + trọng số family | 0,8697 | +0,06 | 325 / 625 |
| b12-hgnn: HGNN + skip + trọng số family | 0,8665 | −0,26 | 290 / 590 |

- **Dừng sớm không phải nguyên nhân của HSL:** HSL đạt best ở epoch 90 rồi 300
  epoch không cải thiện (b0 ≈ lần 3). HGNN thì vẫn tăng chậm tới epoch 840.
- **Khi không có skip, module HSL làm hại:** cùng điều kiện train, HSL thấp hơn
  HGNN 1,35 điểm. Phù hợp với mục 3, 5 và 6 (ΔH vào Behavioral, nhiễu Gumbel,
  contrastive ngược chiều lan truyền).
- **Skip xác nhận nguyên nhân 1 (pha loãng):** +3,7 điểm cho HSL.
- **Khi có skip, graph chưa thêm gì so với MLP** (+0,0008, trong vùng nhiễu), dù
  mục 1 cho thấy graph có thông tin (+0,5 điểm với GBDT). Giả thuyết: do nguyên
  nhân 3 (lệch train/đánh giá), lớp phân loại không tin vào nhánh graph mà dựa vào
  nhánh MLP. Kiểm tra bằng `b1-nob`, `b1-nob-hgnn` (bỏ Behavioral) và bằng lan
  truyền tách rời, trong đó đặc trưng trung bình theo family được tính trước giống
  nhau cho train và đánh giá (SGC, Wu et al., ICML 2019; SIGN, Rossi et al., 2020).
- Khi có skip, HSL hơn HGNN 0,003. Có thể do lấy mẫu Gumbel đóng vai trò chống
  overfit cho nhánh graph (giống DropEdge), nhưng mới có 1 seed nên chưa kết luận.

**Bỏ Behavioral (seed 1):**

| Cấu hình | Val AUC | Test AUC | Test AUPRC | Best / đã chạy |
|---|---|---|---|---|
| b1-nob: HSL + skip, Course + Object | 0,8673 | 0,8692 | 0,9405 | 310 / 610 |
| **b1-nob-hgnn: HGNN + skip, Course + Object** | **0,8710** | **0,8730** | **0,9450** | 745 / 1000 |

- **Lần đầu graph vượt MLP:** +0,0039 test AUC, +0,0038 val AUC, +0,004 AUPRC;
  vượt cả GBDT (+0,0031). Val và test tăng cùng mức, và mức tăng khớp với trần dự
  đoán ở mục 1. Xác nhận nguyên nhân 2 và 3: Behavioral chặn thông tin hữu ích.
- **HSL giờ kéo xuống:** cùng cấu hình, HSL thấp hơn HGNN 0,0038. Không còn
  Behavioral thì ΔH cũng tắt, nên khác biệt chỉ còn mask cứng Me/Mv và
  contrastive, phù hợp với mục 5. Khi còn Behavioral, nhiễu Gumbel có thể đã giúp
  chống lại phần thừa; bỏ Behavioral đi thì lợi ích này mất.
- **Đã xác nhận bằng 5 seed (30/09/2026):** HGNN + skip trên Course + Object đạt
  test AUC 0,8729 ± 0,0001, AUPRC 0,9449 ± 0,0001, AUPRC lớp không bỏ học
  0,7571 ± 0,0001 (MLP: 0,8693 / 0,9410 / 0,7514; GBDT: 0,8699 / 0,9418 / 0,7526).
  Macro-F1 0,7765 thấp hơn MLP (0,7800) một chút vì t\* tối ưu F1 của lớp bỏ học.
- **Ablation b1-hgnn (HGNN + skip, đủ 3 loại, không trọng số family), seed 1:**
  test AUC 0,8665, val 0,8662, best epoch 305/605. So với b1-nob-hgnn (0,8730), riêng
  việc bỏ Behavioral tăng **+0,0065**. Khi còn Behavioral, mô hình graph **thấp hơn
  MLP** 0,0026 dù đã có skip: lớp phân loại dựa vào đặc trưng từ Behavioral lúc
  train, rồi bị hại lúc đánh giá khi đặc trưng đó đổi phân phối (mục 4). b12-hgnn
  (trọng số family bản cũ) cũng đạt 0,8665, xác nhận bản cũ không học được gì.
- **HSL chỉ có lợi khi graph còn hyperedge có hại.** Cùng skip, HSL − HGNN =
  +0,0034 (đủ 3 loại), +0,0032 (đủ 3 loại, trọng số family bản cũ), nhưng −0,0038
  (bỏ Behavioral). Lấy mẫu Gumbel hoạt động như DropEdge, giảm mức dựa vào
  Behavioral, nhưng không phát hiện được nó (`kept` ≈ 0,99). Bỏ Behavioral một cách
  tường minh tốt hơn 0,003 AUC.
- **Trọng số family bản sửa (lr 0,05, không weight decay), seed 1:**

  | HGNN + skip | Đủ 3 loại | Bỏ Behavioral |
  |---|---|---|
  | Không trọng số family | 0,8665 | 0,8730 (5 seed: 0,8729 ± 0,0001) |
  | Trọng số family bản cũ | 0,8665 | – |
  | **Trọng số family bản sửa** | **0,8728** (val 0,8711) | **0,8740** (val 0,8721) |

  Với đủ 3 loại, trọng số family bản sửa tăng +0,0063, gần bằng bỏ Behavioral bằng
  tay: model tự khắc phục tác hại của Behavioral. Khi đã bỏ Behavioral, trọng số
  family vẫn thêm +0,0010 (0,8740, cấu hình tốt nhất đến nay). Macro-F1 của
  fw2-hgnn là 0,7803, ngang MLP (0,7800). Lỗi của bản cũ là do tốc độ học và weight
  decay, không phải do ý tưởng. Còn cần 5 seed cho fw2-hgnn, fw2-nob-hgnn.

- **Trọng số family đã học** (epoch cuối, seed 1; quy về self-loop = 1, vì chỉ tỉ lệ
  có nghĩa: nhân mọi `w` với cùng hệ số thì đầu ra không đổi):

  | Family | fw2-hgnn | fw2-nob-hgnn |
  |---|---|---|
  | Self-loop | 1 (w = 4,25) | 1 (w = 2,14) |
  | Course | 0,213 | 0,216 |
  | Object | 0,132 | 0,126 |
  | Behavioral | **0,013** | không dùng (w giữ đúng 1,000) |

  Model **tự tắt Behavioral** (1/80 so với self-loop) và **tự tăng trọng số của
  chính node khoảng 4 lần**, đúng hai nguyên nhân ở mục 2 và 3. Với một target
  validation điển hình (1 course, 3 object, 1 behavioral, 1 self), phần đóng góp đổi
  từ self 17% / object 50% / course 17% / behavioral 17% sang **self 62% / object
  24% / course 13% / behavioral 0,8%**. Hai run độc lập cho cùng tỉ lệ (Course/self
  0,213 và 0,216; Object/self 0,132 và 0,126), nên trọng số là một phép đo ổn định.
  `w_behavioral` giữ đúng 1,000 khi không có Behavioral, xác nhận optimizer mới
  không còn kéo trọng số bằng weight decay.
- (Ghi chú cũ) Cần bổ sung hai ô ablation `b1-hgnn` (HGNN + skip, đủ
  family, không `W`) và `b12-nob-hgnn` (thêm `W`, bỏ Behavioral) để tách tác dụng
  của việc bỏ Behavioral khỏi tác dụng của trọng số family.

## Tài liệu dẫn

- Feng, Y. et al. *Hypergraph Neural Networks.* AAAI 2019 (ma trận `W`, Eq. 10).
- Cai, D. et al. *Hypergraph Structure Learning for Hypergraph Neural Networks.*
  IJCAI 2022.
- Li, Q., Han, Z., Wu, X.-M. *Deeper Insights into Graph Convolutional Networks for
  Semi-Supervised Learning.* AAAI 2018 (over-smoothing).
- Ding, K. et al. *Be More with Less: Hypergraph Attention Networks for Inductive
  Text Classification.* EMNLP 2020 (HyperGAT).
- Chien, E. et al. *You are AllSet.* ICLR 2022.
- Huang, J., Yang, J. *UniGNN.* IJCAI 2021 (initial residual).
- Radovanović, M., Nanopoulos, A., Ivanović, M. *Hubs in Space: Popular Nearest
  Neighbors in High-Dimensional Data.* JMLR 2010 (hubness của kNN).
- Jang, E., Gu, S., Poole, B. *Categorical Reparameterization with Gumbel-Softmax.*
  ICLR 2017.

Các tài liệu dẫn ở trên theo hiểu biết chung; cần mở bản gốc để kiểm tra trước khi
đưa vào bài.
