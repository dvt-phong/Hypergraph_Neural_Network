# Nhận xét báo cáo "Mô hình Hypergraph Neural Network" (version 1)

Đối chiếu với code trong `src/` (commit `9b1bd5f`), dữ liệu đã xử lý trong
`data/processed/simple/` và kết quả 5 seed ghi trong `docs/ket_qua_thi_nghiem.xlsx`.

Cấu trúc tài liệu:

- **Phần A.** Kết luận nhanh
- **Phần B.** Đối chiếu từng mục của báo cáo với code
- **Phần C.** Những nội dung còn thiếu
- **Phần D.** Góp ý cho hình kiến trúc
- **Phần E.** Gợi ý viết lại mục 1–4
- **Phần F.** Mục 5 hoàn chỉnh: Các siêu tham số
- **Phần G.** Mục 6 hoàn chỉnh: Kết quả ban đầu
- **Phần H.** Lỗi chính tả và trình bày

---

## A. Kết luận nhanh

Khung báo cáo đúng với pipeline thật: preprocess → feature → H0 → HGNN → HSL →
BCE + contrastive. Tuy nhiên có **4 chỗ mô tả sai so với code** và **3 phần quan
trọng còn thiếu**.

**Sai so với code (cần sửa):**

1. Hình vẽ có **User hyperedge**, nhưng code chưa có. Code dùng Course, Object,
   Behavioral và self-loop.
2. Mục 2 ghi `user_id`, `course_id`, `start`, `end` là feature. Thực tế chúng
   **không nằm trong X**. Chỉ có gender, education, tuổi (và cờ thiếu tuổi), category.
3. Mục 2 ghi birth thiếu được điền bằng **trung bình**. Code điền bằng **trung
   vị** của tập train, rồi thêm cột cờ `age_missing`.
4. Mục 4 mô tả HSL như **tăng cường dữ liệu ngẫu nhiên**, dẫn theo DP-SCL. Code
   cài đặt **HSL** (Cai et al., IJCAI 2022). Việc giữ hay bỏ hyperedge/node do
   **MLP học được** quyết định, không phải ngẫu nhiên. Bước thêm node chỉ áp dụng
   cho **Behavioral hyperedge**.

**Còn thiếu (nên bổ sung):**

1. Cách **suy luận cho validation/test**: mỗi enrollment cần dự đoán có một
   hypergraph cục bộ riêng, không đưa vào H0. Đây là điểm mạnh về chống rò rỉ dữ
   liệu (leakage) nhưng báo cáo chưa nhắc tới.
2. Công thức HGNN, cấu trúc encoder, cách tính contrastive loss (cặp dương, cặp âm).
3. Số liệu cụ thể: kích thước từng split, tỉ lệ dropout, thống kê hypergraph.

---

## B. Đối chiếu từng mục

Ký hiệu: ✅ đúng · ⚠️ đúng nhưng chưa đủ hoặc chưa chính xác · ❌ sai so với code.

### B.1. Mục 1: Dataset

| Báo cáo viết | Thực tế trong code/dữ liệu | Đánh giá |
|---|---|---|
| 225.642 enrollment | 126.354 (train) + 31.589 (validation) + 67.699 (test) = 225.642 | ✅ |
| 77.083 người học | Đếm lại trên 3 split: 77.083 (train 63.277, validation 24.932, test 44.008; các tập có chung người học vì chia theo enrollment) | ✅ |
| 247 khóa học | 247 course hyperedge trong train, mọi khóa ở test đều có trong train | ✅ |
| 23 hoạt động, chia 4 nhóm 5/6/5/7 | `ACTION_GROUPS` trong [0_config.py:32-48](../src/0_config.py#L32-L48). Action nằm ngoài 23 loại này bị bỏ | ✅ |
| Thời gian theo dõi 35 ngày | Cửa sổ quan sát là **ngày 0 đến 34 tính từ ngày bắt đầu khóa học** (`course_start`), không phải từ ngày ghi danh ([2_preprocess.py:126-129](../src/2_preprocess.py#L126-L129)) | ⚠️ nên ghi rõ mốc |
| train_log + train_truth → train và validation | Chia ngẫu nhiên **theo enrollment**, tỉ lệ **80:20**, seed 1 ([2_preprocess.py:64-75](../src/2_preprocess.py#L64-L75)) | ⚠️ thiếu tỉ lệ và cách chia |
| test_log + test_truth → test | Giữ nguyên tập test chính thức | ✅ |
| (chưa có) | Nhãn: `truth = 1` là **bỏ học**. Tỉ lệ bỏ học 75,8% ở cả 3 tập | ⚠️ thiếu |

### B.2. Mục 2: Feature Engineering

| Báo cáo viết | Thực tế trong code | Đánh giá |
|---|---|---|
| Preprocess tạo `train.csv`, `validation.csv`, `test.csv` | Đúng. Mỗi dòng là **một sự kiện** đã ghép thông tin user và course | ✅ |
| (chưa có) | Enrollment không có sự kiện nào trong 35 ngày **vẫn được giữ**, vector hành vi bằng 0: 1.879 train (1,49%), 472 validation, 954 test | ⚠️ thiếu |
| 35 daily action counts | 35 cột, mỗi cột là tổng số sự kiện trong ngày đó | ✅ |
| 23 cột đếm theo sự kiện | Đúng | ✅ |
| "Chỉ đếm số lần thực hiện" | Sau khi đếm, 58 cột hành vi được biến đổi `log(1 + x)` rồi **chuẩn hóa z-score** bằng mean/std của **tập train** ([3_features.py:234-251](../src/3_features.py#L234-L251)) | ❌ thiếu bước chuẩn hóa |
| User: `user_id`, gender, education, birth | `user_id` **không phải feature**. Feature user gồm: gender (4 cột), education (9 cột), tuổi (1), cờ thiếu tuổi (1) = 15 | ❌ |
| Course: `course_id`, start, end, category | Chỉ **category** là feature (17 + missing + other = 19 cột). `course_id` dùng để dựng Course hyperedge. `start` dùng để tính ngày của sự kiện và tuổi. `end` **không dùng** | ❌ |
| Tuổi = năm bắt đầu khóa − năm sinh | Đúng. Tuổi ngoài khoảng [10, 100] bị coi là thiếu | ⚠️ thiếu điều kiện hợp lệ |
| Birth thiếu → điền **trung bình** | Điền **trung vị** của train, sau đó z-score theo train, và có thêm cột `age_missing` | ❌ |
| One-hot có thêm missing, other | Đúng | ✅ |
| 58 / 15 / 19 feature | Đúng. Tổng **92** chiều (báo cáo chưa ghi tổng) | ✅ |
| `train_X.npy`, `test_X.npy`, `validation_X.npy` | Tên thật: `train/X.npy`, `validation/X.npy`, `test/X.npy`, kèm `feature_names.csv` | ⚠️ nhỏ |
| (chưa có) | Mọi thống kê chuẩn hóa **chỉ tính trên train**, rồi áp cho validation/test | ⚠️ thiếu, nhưng là điểm cộng nên ghi |

### B.3. Mục 3: Hypergraph Construction

| Báo cáo viết | Thực tế trong code | Đánh giá |
|---|---|---|
| "Mô hình sử dụng 3 hyperedge" | Là 3 **loại** (family) hyperedge, mỗi loại gồm nhiều hyperedge. Ngoài ra mỗi node có một **self-loop hyperedge** không bao giờ bị HSL bỏ | ⚠️ |
| Course hyperedge: cùng khóa học | Đúng. 247 hyperedge, kích thước 151 / 377 / 2.361 (min / trung vị / max) | ✅ |
| Object hyperedge: cùng khóa, cùng object | Đúng. Object chỉ gồm **video, problem, forum**. Nhóm web page không có mã object nên không tạo hyperedge. Khóa của hyperedge là `course_id | loại | object_id`. 21.747 hyperedge, kích thước 2 / 27 / 1.893 | ⚠️ thiếu chi tiết |
| Behavior: kNN, 10 láng giềng, cosine | Đúng. kNN chỉ dùng **58 cột hành vi** (không dùng user/course), không tính chính nó. Mỗi hyperedge = 1 node tâm + 10 láng giềng = 11 node. 126.354 hyperedge | ⚠️ thiếu chi tiết |
| (chưa có) | Láng giềng thứ 11–20 (`k_max = 20`) được lưu làm **ứng viên** cho bước thêm node của HSL | ⚠️ thiếu |
| (chưa có) | Hyperedge có ít hơn 2 thành viên bị loại | ⚠️ nhỏ |
| Dựng ma trận H0 | Đúng. H0 **chỉ chứa node train** (126.354 node, 3.201.735 liên thuộc kể cả self-loop) | ⚠️ thiếu điều kiện "chỉ train" |
| Đưa H0 qua HGNN thu được Z0 | Đúng. HGNN 2 lớp, chiều ẩn 128 | ✅ |

### B.4. Mục 4: Hypergraph Structure Learning

| Báo cáo viết | Thực tế trong code | Đánh giá |
|---|---|---|
| "Tăng cường dữ liệu, tương tự học tương phản trong DP-SCL" | Code cài đặt **HSL** (Cai et al., IJCAI 2022): `H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I`. Đây là **học cấu trúc có tham số**, không phải augmentation ngẫu nhiên. Nếu muốn nhắc DP-SCL thì chỉ nên nói ý tưởng dùng contrastive loss giống nhau, còn 3 thao tác trên cấu trúc phải dẫn nguồn HSL | ❌ |
| Giải pháp 1: bỏ hẳn **1** hyperedge | **Hyperedge sampling (Me)**: *mỗi* hyperedge có xác suất giữ `σ(MLP([h_e ‖ loại hyperedge]))`, nên có thể bỏ nhiều hyperedge. Self-loop luôn được giữ | ❌ |
| Giải pháp 2: thêm 2 node vào hyperedge | **Implicit connections (ΔH)**: chỉ áp dụng cho **Behavioral hyperedge**. Chọn 2 node trong 10 ứng viên (láng giềng 11–20) có cosine(`z_v`, `h_e`) cao nhất, không truyền gradient | ⚠️ thiếu điều kiện |
| Giải pháp 3: bỏ node khỏi hyperedge | **Incident node sampling (Mv)**: mỗi cặp (node, hyperedge) có xác suất giữ `σ(MLP([z_v ‖ h_e]))`, áp dụng cả cho node vừa thêm ở ΔH | ⚠️ thiếu cơ chế |
| Áp dụng 3 giải pháp cùng lúc | Đúng, theo thứ tự: thêm ΔH trước, rồi mới lấy mẫu Me và Mv | ⚠️ thiếu thứ tự |
| (chưa có) | Khi train, mask lấy mẫu bằng **Gumbel straight-through** (τ = 0,4): chiều xuôi là 0/1, chiều ngược có gradient. Khi đánh giá, giữ nếu xác suất > 0,5 | ⚠️ thiếu |
| (chưa có) | `h_e` = trung bình Z0 của các thành viên trong H0. Mọi mask được tính từ **Z0**, một lần mỗi epoch. H* được **tạo lại từ H0 mỗi epoch**, không cộng dồn | ⚠️ thiếu |
| Đưa H* qua HGNN thu được Z* | Đúng. HGNN cho Z* **dùng chung trọng số** với HGNN cho Z0 | ⚠️ nên ghi "dùng chung trọng số" |
| Z0 + Z* → contrastive; Z* → BCE | Đúng | ✅ |
| (chưa có) | Contrastive: cặp dương là cùng một node ở hai view. Cặp âm là 32 node **cùng hyperedge** (lấy mẫu), lấy từ cả hai view. 1.024 node neo mỗi epoch, τ = 0,07, tính đối xứng hai chiều | ⚠️ thiếu |
| (chưa có) | BCE có trọng số `pos_weight = #không bỏ học / #bỏ học = 0,319` | ⚠️ thiếu |
| Total loss | `L = L_BCE + λ · L_CL`, λ = 0,1 | ✅ (nên ghi giá trị λ) |

---

## C. Những nội dung còn thiếu

### C.1. Suy luận cho validation/test (quan trọng nhất)

Báo cáo chưa nói node validation/test được đưa vào mô hình như thế nào. Code làm
như sau ([4_hypergraph.py:263-305](../src/4_hypergraph.py#L263-L305)):

- H0 **chỉ chứa node train**. Không node validation/test nào tham gia lúc train.
- Với mỗi enrollment cần dự đoán `t`, dựng một **hypergraph cục bộ** gồm `t` và các
  node train liên quan:
  - Course hyperedge của khóa `t` học (chỉ gồm thành viên train);
  - Object hyperedge của mỗi object mà `t` đã tương tác;
  - Behavioral hyperedge của `t`: 10 láng giềng gần nhất **trong tập train**;
  - self-loop cho mọi node.
- Chạy mô hình (HGNN + HSL, mask tất định, không dropout) trên đồ thị cục bộ đó.
  Chỉ lấy đầu ra tại node `t`.
- Nhiều đồ thị cục bộ được ghép thành một khối chéo để chạy nhanh. Không hyperedge
  nào nối hai target với nhau, nên các target không "nhìn thấy" nhau.

Đây là thiết lập **quy nạp (inductive)**. Nó chặt chẽ hơn cách nhiều bài GNN dự
báo bỏ học đang làm (đưa luôn node test vào đồ thị khi train). Nên viết thành một
mục riêng, ví dụ **3.4. Suy luận cho enrollment mới**.

### C.2. Kiến trúc HGNN

Nên ghi công thức lan truyền (Feng et al., AAAI 2019):

```
X' = Dv^(-1/2) · H · De^(-1) · Hᵀ · Dv^(-1/2) · (X·Θ + b)
```

Encoder gồm 2 lớp: `Linear → lan truyền → ReLU → Dropout → Linear → lan truyền → ReLU`.
Classifier là một lớp `Linear(128 → 1)` + sigmoid. Code dùng ma trận trọng số
hyperedge `W = I` (mọi hyperedge có trọng số như nhau).

### C.3. Quy trình huấn luyện và đánh giá

- Full-batch: mỗi epoch là **một** bước cập nhật trên toàn bộ hypergraph train.
- Cứ 5 epoch đánh giá trên validation một lần. Giữ checkpoint có **AUC validation**
  cao nhất. Dừng sớm sau 5 lần đánh giá liên tiếp không cải thiện.
- Tập test chỉ chạy **một lần** trên checkpoint đã chọn.
- Chỉ số: AUC, AUPRC, F1, Precision, Recall. Lớp dương là "bỏ học", ngưỡng 0,5.
- Lặp lại với 5 seed: 1, 11, 111, 1111, 11111. Báo cáo mean ± std.

### C.4. Bảng thống kê hypergraph

Nên thêm bảng này vào cuối mục 3 (số liệu đo từ `hypergraph.npz`):

| Loại hyperedge | Số hyperedge | Kích thước min / trung vị / max | Số liên thuộc |
|---|---:|---|---:|
| Course | 247 | 151 / 377 / 2.361 | 126.354 |
| Object | 21.747 | 2 / 27 / 1.893 | 1.559.133 |
| Behavioral | 126.354 | 11 / 11 / 11 | 1.389.894 |
| Self-loop | 126.354 | 1 | 126.354 |
| **Tổng** | 274.702 | | **3.201.735** |

---

## D. Góp ý cho hình kiến trúc

1. **Bỏ "User hyperedge"** khỏi khối Hypergraph Construction. Nếu muốn giữ vì đây
   là hướng sắp làm (H5 trong `IMPROVEMENT_PLAN.md`), vẽ bằng **nét đứt** và ghi
   chú "(dự kiến)".
2. **Thêm "Self-loop"**, hoặc ghi chú trong chú thích hình rằng mỗi node có một
   self-loop.
3. Trong khối HSL, **thiếu bước thêm node (ΔH)**. Nên có 3 ô theo đúng thứ tự code:
   `Implicit Connections (ΔH)` → `Hyperedge Sampling (Me)` → `Incident Node Sampling (Mv)` → `Refined H*`.
4. Vòng lặp nét đứt "Hyperedge representations / Node representations" dễ gây hiểu
   lầm là H* được tinh chỉnh lặp nhiều lần. Trong code, `z_v` và `h_e` lấy từ **Z0**,
   tính một lần. Nên nối mũi tên từ `Embedding Z0` vào khối HSL và ghi
   "z_v, h_e từ Z0".
5. Ghi **"shared weights"** giữa hai khối HGNN.
6. Thêm tên hình, ví dụ *Hình 1. Kiến trúc tổng quan của mô hình HGNN + HSL*. Hiện
   hình nằm trước mục 1 mà chưa có tiêu đề.
7. (Tùy chọn) Thêm một nhánh nhỏ cho suy luận: `Validation/Test → Local hypergraph → Model → ŷ`.

---

## E. Gợi ý viết lại mục 1–4

Phần dưới là bản đề xuất, em có thể chép vào báo cáo và chỉnh lại văn phong.

### Mục 1. Dataset (thêm sau bảng)

> - Nhãn: `truth = 1` nghĩa là người học bỏ học. Tỉ lệ bỏ học khoảng 75,8%, gần
>   như bằng nhau ở cả ba tập.
> - Cửa sổ quan sát: chỉ dùng các sự kiện trong 35 ngày đầu, tính từ ngày bắt đầu
>   khóa học (ngày 0 đến ngày 34).
> - Chia dữ liệu: tập test giữ nguyên theo tập test chính thức. Tập train chính
>   thức (157.943 enrollment) được chia ngẫu nhiên theo enrollment, tỉ lệ 80:20
>   (seed 1), thành train và validation.
>
> | Tập | Số enrollment | Tỉ lệ bỏ học |
> |---|---:|---:|
> | Train | 126.354 | 75,8% |
> | Validation | 31.589 | 75,8% |
> | Test | 67.699 | 75,8% |

### Mục 2. Feature Engineering (thay đoạn "Thông tin user / course" và đoạn điền birth)

> Mỗi enrollment được biểu diễn bằng vectơ 92 chiều gồm ba nhóm:
>
> - **Hành vi (58 chiều):** 35 cột số sự kiện theo từng ngày, và 23 cột số lần
>   thực hiện mỗi loại hành động. Các số đếm được biến đổi `log(1 + x)` rồi chuẩn
>   hóa z-score.
> - **Người học (15 chiều):** giới tính one-hot (4 cột: female, male, missing,
>   other); trình độ học vấn one-hot (9 cột: 7 mức, missing, other); tuổi khi bắt
>   đầu khóa học (1 cột); cờ thiếu tuổi (1 cột). Tuổi bằng năm bắt đầu khóa trừ
>   năm sinh. Tuổi ngoài khoảng [10, 100] được coi là thiếu và được điền bằng
>   trung vị, sau đó chuẩn hóa z-score.
> - **Khóa học (19 chiều):** lĩnh vực khóa học one-hot (17 lĩnh vực, missing, other).
>
> `user_id` và `course_id` không dùng làm đặc trưng. `course_id` được dùng để dựng
> Course hyperedge (mục 3). Mọi thống kê (mean, std, trung vị) chỉ ước lượng trên
> tập train, sau đó áp dụng nguyên cho validation và test để tránh rò rỉ dữ liệu.
> Các enrollment không có sự kiện nào trong cửa sổ quan sát (khoảng 1,5%) vẫn được
> giữ, với phần hành vi bằng 0.

### Mục 3. Hypergraph Construction (thay đoạn mô tả hyperedge)

> Hypergraph ban đầu H0 chỉ gồm các enrollment của tập train. Mỗi enrollment là
> một node. Có ba loại hyperedge:
>
> - **Course hyperedge:** tất cả enrollment cùng một khóa học.
> - **Object hyperedge:** tất cả enrollment cùng khóa đã tương tác với cùng một học
>   liệu (video, bài tập hoặc forum). Nhóm web page không có mã học liệu nên không
>   tạo loại hyperedge này.
> - **Behavioral hyperedge:** mỗi enrollment `v` là tâm của một hyperedge gồm `v`
>   và 10 enrollment train có hành vi gần nhất (cosine trên 58 chiều hành vi). Các
>   láng giềng thứ 11 đến 20 được giữ lại làm ứng viên cho bước học cấu trúc.
>
> Ngoài ra, mỗi node có một self-loop hyperedge để không node nào bị cô lập. H0
> được đưa qua bộ mã hóa HGNN hai lớp để thu được biểu diễn Z0.

### Mục 4. Hypergraph Structure Learning (viết lại toàn bộ)

> Hypergraph H0 được dựng theo quy tắc cố định nên có thể chứa liên kết nhiễu (ví
> dụ hai người học cùng khóa nhưng hành vi rất khác) và thiếu liên kết hữu ích.
> Theo phương pháp HSL (Cai et al., IJCAI 2022), mô hình học một hypergraph tinh
> chỉnh:
>
> **H\* = Me ⊙ Mv ⊙ (H0 + ΔH) + I**
>
> Các thành phần được tính từ Z0. Biểu diễn của hyperedge `h_e` là trung bình Z0
> của các thành viên:
>
> 1. **Thêm liên kết tiềm ẩn (ΔH):** với mỗi Behavioral hyperedge, chọn 2 node
>    trong số 10 ứng viên có độ tương đồng cosine với `h_e` cao nhất và thêm vào
>    hyperedge. Chỉ áp dụng cho Behavioral hyperedge, vì thêm người học khóa khác
>    vào Course hay Object hyperedge sẽ làm sai ý nghĩa của các quan hệ đó.
> 2. **Lấy mẫu hyperedge (Me):** mỗi hyperedge được giữ với xác suất
>    `σ(MLP([h_e ‖ loại hyperedge]))`.
> 3. **Lấy mẫu node trong hyperedge (Mv):** mỗi cặp (node, hyperedge), kể cả cặp
>    vừa thêm ở bước 1, được giữ với xác suất `σ(MLP([z_v ‖ h_e]))`.
>
> Khi huấn luyện, các mask được lấy mẫu bằng Gumbel straight-through: giá trị là 0/1
> nhưng vẫn truyền được gradient. Khi đánh giá, một thành phần được giữ nếu xác
> suất lớn hơn 0,5. Self-loop luôn được giữ. H\* được đưa qua **cùng** bộ mã hóa
> HGNN (dùng chung trọng số) để thu được Z\*.
>
> Hàm mất mát gồm hai phần:
>
> - **BCE có trọng số** trên dự đoán từ Z\*, với trọng số lớp dương
>   `pos_weight = N_không bỏ học / N_bỏ học`.
> - **Contrastive loss trong hyperedge** giữa Z0 và Z\*: cặp dương là cùng một node
>   ở hai view. Cặp âm là các node cùng hyperedge với node đó, lấy từ cả hai view.
>   Loss này giữ cho H\* không lệch quá xa H0 và giúp các node cùng khóa vẫn phân
>   biệt được với nhau, tránh over-smoothing.
>
> **L_total = L_BCE + λ · L_CL**, với λ = 0,1.

Sau đó thêm mục mới **"Suy luận cho validation/test"** theo nội dung ở phần C.1.

---

## F. Mục 5 hoàn chỉnh: Các siêu tham số

> **5. Các siêu tham số**
>
> Bảng 5 liệt kê cấu hình dùng cho kết quả ở mục 6. Đây là cấu hình mặc định, chưa
> qua tinh chỉnh siêu tham số.
>
> **Bảng 5. Siêu tham số của mô hình.**
>
> | Nhóm | Siêu tham số | Giá trị |
> |---|---|---|
> | Dữ liệu | Cửa sổ quan sát | 35 ngày |
> | | Tỉ lệ train : validation | 80 : 20 (seed chia dữ liệu = 1) |
> | | Số chiều đặc trưng | 92 (58 hành vi + 15 người học + 19 khóa học) |
> | Hypergraph | Số láng giềng kNN của Behavioral hyperedge (k) | 10 |
> | | Số láng giềng tối đa, dùng làm ứng viên ΔH (k_max) | 20 |
> | | Độ đo tương đồng | Cosine trên 58 chiều hành vi |
> | Bộ mã hóa HGNN | Số lớp | 2 |
> | | Số chiều ẩn | 128 |
> | | Dropout | 0,3 |
> | HSL | Số node thêm vào mỗi Behavioral hyperedge | 2 |
> | | Nhiệt độ Gumbel | 0,4 |
> | | Số chiều ẩn của MLP chấm điểm (Me, Mv) | 32 |
> | | Xác suất giữ ban đầu | σ(3) ≈ 0,95 |
> | Contrastive loss | Hệ số λ | 0,1 |
> | | Nhiệt độ τ | 0,07 |
> | | Số node neo mỗi epoch | 1.024 |
> | | Số mẫu âm mỗi node neo | 32 |
> | Loss phân loại | BCE, trọng số lớp dương | N_âm / N_dương ≈ 0,319 |
> | Tối ưu | Optimizer | Adam |
> | | Learning rate / weight decay | 1e-3 / 5e-4 |
> | | Cắt chuẩn gradient | 5 |
> | | Kích thước batch | Full-batch (1 bước cập nhật mỗi epoch) |
> | | Số epoch tối đa | 200 |
> | | Đánh giá trên validation | Mỗi 5 epoch |
> | | Dừng sớm (patience) | 5 lần đánh giá không cải thiện AUC |
> | | Tiêu chí chọn checkpoint | AUC validation cao nhất |
> | Đánh giá | Ngưỡng phân loại | 0,5 |
> | | Seed | 1, 11, 111, 1111, 11111 |

Nguồn giá trị: `DEFAULT_SETTINGS` trong [8_train.py:38-59](../src/8_train.py#L38-L59),
`StructureLearner` trong [6_hsl.py:51-60](../src/6_hsl.py#L51-L60), `k`, `k_max`
trong [4_hypergraph.py:336-337](../src/4_hypergraph.py#L336-L337).

---

## G. Mục 6 hoàn chỉnh: Kết quả ban đầu

> **6. Kết quả ban đầu**
>
> **6.1. Thiết lập.** Mô hình được huấn luyện với cấu hình ở Bảng 5 và chạy lặp
> lại với 5 seed. Checkpoint được chọn theo AUC trên tập validation. Tập test chỉ
> được đánh giá một lần trên checkpoint đã chọn. Lớp dương là "bỏ học", ngưỡng
> phân loại 0,5. Kết quả báo dưới dạng trung bình ± độ lệch chuẩn qua 5 seed.
>
> Để có mốc so sánh, bảng có thêm một bộ dự đoán đơn giản: dự đoán mọi enrollment
> đều bỏ học. Với tỉ lệ bỏ học 75,8%, bộ dự đoán này có Precision = 0,758,
> Recall = 1, nên F1 = 0,862 và AUPRC = 0,758.
>
> **Bảng 6. Kết quả trên tập test (5 seed).**
>
> | Mô hình | AUC | AUPRC | F1 |
> |---|---|---|---|
> | Dự đoán "tất cả bỏ học" | 0,500 | 0,758 | 0,862 |
> | HGNN + HSL (đề xuất) | 0,809 ± 0,003 | 0,910 ± 0,002 | 0,44 ± 0,30 |
>
> **6.2. Nhận xét.**
>
> - **Khả năng xếp hạng ổn định.** AUC đạt 0,809 với độ lệch chuẩn chỉ 0,003 giữa
>   các seed. AUPRC đạt 0,910, cao hơn mức ngẫu nhiên (0,758) khoảng 0,15. Như vậy
>   mô hình phân biệt được người bỏ học và người tiếp tục học, và kết quả này không
>   phụ thuộc nhiều vào khởi tạo.
> - **F1 thấp và dao động mạnh.** F1 chỉ đạt 0,44 ± 0,30, thấp hơn cả bộ dự đoán
>   "tất cả bỏ học". Recall trên test dao động từ 0,03 đến 0,68 giữa các seed, trong
>   khi AUC gần như không đổi. Vậy vấn đề nằm ở **ngưỡng 0,5**, không nằm ở khả năng
>   xếp hạng. Có hai nguyên nhân:
>   (i) lớp bỏ học là lớp đa số (75,8%) nên `pos_weight = 0,319` làm giảm trọng số
>   của chính lớp này, kéo xác suất dự đoán xuống dưới 0,5;
>   (ii) checkpoint tốt nhất rơi vào epoch 10–20, tức mô hình mới được cập nhật
>   10–20 lần (full-batch), nên xác suất đầu ra còn dồn quanh 0,5.
> - **Mô hình dừng huấn luyện sớm.** Quá trình dừng ở epoch 35–45 trên tối đa 200.
>   Điều này cho thấy mô hình có thể chưa được huấn luyện đủ.
> - **Tham khảo tài liệu.** Trên dữ liệu XuetangX với cùng cửa sổ 35 ngày, CFIN
>   (Feng, Tang & Liu, AAAI 2019) báo AUC 82,23% cho Logistic Regression và 86,40%
>   cho CFIN. Tập dữ liệu và cách chia của bài đó khác với báo cáo này, nên các con
>   số chỉ mang tính tham khảo, không so sánh trực tiếp được. Tuy nhiên chúng gợi ý
>   rằng bộ đặc trưng hiện tại còn nhiều chỗ để cải thiện.
>
> **6.3. Hạn chế và bước tiếp theo.**
>
> Đây là kết quả ban đầu. Hiện chưa có baseline chạy trên cùng cách chia dữ liệu
> (Logistic Regression, LightGBM, MLP, HGNN không có HSL) và chưa có ablation cho
> từng thành phần của HSL. Vì vậy chưa thể kết luận hypergraph hay HSL đóng góp bao
> nhiêu vào kết quả. Các bước tiếp theo:
>
> 1. Chọn ngưỡng phân loại tối ưu F1 trên validation rồi áp cho test; thử
>    `pos_weight = 1`; báo thêm macro-F1.
> 2. Chạy baseline và ablation: `--no-hsl`, `--no-edge-sampling`,
>    `--no-node-sampling`, `--add-per-edge 0`, `--lambda-cl 0`, và các tập đặc
>    trưng `behavior` / `behavior_user` / `behavior_course`.
> 3. Tăng số bước cập nhật: tăng patience, thử learning rate lớn hơn.
> 4. Bổ sung đặc trưng (session, số ngày hoạt động, tỉ lệ làm bài đúng) và thêm
>    User hyperedge.

### Việc em cần làm trước khi nộp mục 6

Số liệu ở Bảng 6 lấy từ `docs/ket_qua_thi_nghiem.xlsx` (sheet `Ket_qua`, lần chạy
`lan_1`). Sheet `Du_lieu_seed` hiện **đang trống** và máy local không có các file
`*_test.json`. Vì vậy mean ± std của Precision/Recall và số liệu từng seed chưa có.
Nên chép thư mục kết quả từ server về rồi chạy:

```bash
python scripts/collect_results.py result/<dd-mm-yyyy_HH-MM>
python scripts/export_excel.py result/<dd-mm-yyyy_HH-MM> --note "HSL đầy đủ"
```

Sau đó có thể thêm vào mục 6 một bảng theo từng seed:

| Seed | Best epoch | AUC | AUPRC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | |
| 11 | | | | | | |
| 111 | | | | | | |
| 1111 | | | | | | |
| 11111 | | | | | | |
| **Mean ± std** | | 0,809 ± 0,003 | 0,910 ± 0,002 | 0,44 ± 0,30 | | |

Nên ghi thêm tỉ lệ `kept_course`, `kept_object`, `kept_behavioral` ở best epoch
(có trong `history` của file `*_train.json`). Nếu các tỉ lệ này đều khoảng 0,95
(bằng giá trị khởi tạo) thì H\* ≈ H0, tức HSL gần như chưa cắt gì. Đây là điều cần
biết trước khi thầy hỏi "HSL học được gì".

---

## H. Lỗi chính tả và trình bày

| Vị trí | Hiện tại | Sửa thành |
|---|---|---|
| Mục 2, danh sách feature | Fearture | Feature |
| Mục 2, one-hot | "one hot", "hoạt dữ liệu không khớp" | "one-hot", "hoặc dữ liệu không khớp" |
| Mục 3 | Behavior Similitary, cosine similitary | Behavioral Similarity, cosine similarity |
| Mục 3 | hyperdege | hyperedge |
| Mục 4 | hyperede | hyperedge |
| Mục 4 | "Thêm 2 nodes" | "Thêm 2 node" (tiếng Việt không chia số nhiều) |
| Mục 1 | Chi tiết dataset: số dùng dấu phẩy (225,642) | Văn bản tiếng Việt nên dùng dấu chấm (225.642), hoặc thống nhất một kiểu trong cả báo cáo |
| Chung | Thuật ngữ lúc Anh lúc Việt | Lần đầu dùng ghi cả hai, ví dụ "siêu cạnh (hyperedge)", sau đó dùng thống nhất |
| Chung | Chưa đánh số hình, bảng, công thức | Thêm "Hình 1", "Bảng 1…", đánh số công thức |
| Chung | Chưa có tài liệu tham khảo | Thêm ít nhất: Feng et al. 2019 (HGNN), Cai et al. 2022 (HSL), Feng, Tang & Liu 2019 (CFIN, nguồn dữ liệu XuetangX) |

Có thể xem bản trình bày chi tiết hơn của phương pháp trong
[METHOD.md](METHOD.md). Tài liệu đó đã khớp với code và có thể dùng làm nguồn khi
viết version 2.
