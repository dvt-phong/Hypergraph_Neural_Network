# 3. Phương pháp đề xuất

Phần này trình bày mô hình dự báo bỏ học trên MOOC dựa trên mạng nơ-ron siêu đồ thị
có học cấu trúc (Hypergraph Structure Learning, HSL). Mô hình gồm năm thành phần:
(i) biểu diễn đặc trưng cho mỗi lượt ghi danh; (ii) xây dựng siêu đồ thị ban đầu
$\mathcal{H}_0$ từ ba loại quan hệ giữa các lượt ghi danh; (iii) bộ mã hóa HGNN;
(iv) mô-đun học cấu trúc sinh ra siêu đồ thị tinh chỉnh $\mathcal{H}^*$; và
(v) hàm mục tiêu kết hợp phân loại có trọng số với học tương phản trong siêu cạnh.
Cuối cùng, chúng tôi mô tả giao thức suy luận quy nạp nhằm loại bỏ rò rỉ thông tin
từ tập kiểm định và tập kiểm tra.

---

## 3.1. Phát biểu bài toán

Gọi $\mathcal{V}$ là tập các lượt ghi danh (enrollment). Mỗi lượt ghi danh
$v = (u, c) \in \mathcal{V}$ là một cặp học viên $u$ và khóa học $c$, có nhãn
$y_v \in \{0, 1\}$, trong đó $y_v = 1$ nghĩa là học viên bỏ học. Tập $\mathcal{V}$
được chia thành ba tập rời nhau
$\mathcal{V} = \mathcal{V}_{tr} \cup \mathcal{V}_{val} \cup \mathcal{V}_{te}$.
$\mathcal{V}_{te}$ là tập kiểm tra chính thức của bộ dữ liệu. $\mathcal{V}_{tr}$ và
$\mathcal{V}_{val}$ được tách ngẫu nhiên theo tỉ lệ 80:20 từ tập huấn luyện chính thức.

Với mỗi $v$, mô hình chỉ quan sát nhật ký hoạt động trong cửa sổ $T = 35$ ngày đầu
tiên kể từ ngày khóa học bắt đầu. Nếu $t_0(c)$ là ngày bắt đầu khóa $c$, một sự kiện
tại thời điểm $t$ được giữ lại khi và chỉ khi

$$
0 \le d = \lfloor t - t_0(c) \rfloor_{\text{ngày}} < T. \tag{1}
$$

Mục tiêu là học hàm $f$ sao cho $\hat{y}_v = f(v \mid \mathcal{V}_{tr}) \approx y_v$
với mọi $v \in \mathcal{V}_{val} \cup \mathcal{V}_{te}$. Trong đó $f$ chỉ được phép
dùng đặc trưng và quan hệ của các lượt ghi danh huấn luyện $\mathcal{V}_{tr}$, không
được dùng thông tin của các lượt ghi danh cần dự báo khác.

**Bảng 1. Ký hiệu chính.**

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathbf{x}_v \in \mathbb{R}^{92}$ | Vectơ đặc trưng của lượt ghi danh $v$ |
| $\mathbf{H}_0 \in \{0,1\}^{N \times M}$ | Ma trận liên thuộc của siêu đồ thị ban đầu |
| $\Delta\mathbf{H}$ | Các liên thuộc tiềm ẩn được bổ sung |
| $\mathbf{M}_e, \mathbf{M}_v$ | Mặt nạ lấy mẫu siêu cạnh và lấy mẫu nút–siêu cạnh |
| $\mathbf{H}^*$ | Ma trận liên thuộc sau khi học cấu trúc |
| $\mathbf{Z}^0, \mathbf{Z}^*$ | Biểu diễn nút trên $\mathbf{H}_0$ và trên $\mathbf{H}^*$ |
| $\mathbf{h}_e$ | Biểu diễn của siêu cạnh $e$ |
| $f_\theta$ | Bộ mã hóa HGNN (dùng chung cho hai góc nhìn) |

---

## 3.2. Biểu diễn đặc trưng nút

Mỗi lượt ghi danh $v$ được biểu diễn bằng vectơ ghép bốn khối

$$
\mathbf{x}_v = \big[\, \tilde{\mathbf{b}}^{\text{day}}_v \,\|\, \tilde{\mathbf{b}}^{\text{act}}_v \,\|\, \mathbf{u}_v \,\|\, \mathbf{c}_v \,\big] \in \mathbb{R}^{35 + 23 + 15 + 19}. \tag{2}
$$

**Đặc trưng hành vi theo thời gian.** $\mathbf{b}^{\text{day}}_v \in \mathbb{N}^{T}$,
trong đó $b^{\text{day}}_v[d]$ là số sự kiện hợp lệ của $v$ trong ngày $d$ của khóa học.

**Đặc trưng hành vi theo loại hành động.** $\mathbf{b}^{\text{act}}_v \in \mathbb{N}^{|\mathcal{A}|}$,
trong đó $b^{\text{act}}_v[a]$ là tổng số lần thực hiện hành động $a$ trong cửa sổ quan sát.
Tập $\mathcal{A}$ gồm 23 hành động thuộc bốn nhóm: *video* (5), *assignment* (6),
*forum* (5) và *web page* (7).

Hai khối hành vi được làm mượt bằng $\log(1+\cdot)$ để giảm độ lệch của phân phối số
đếm, sau đó chuẩn hóa z-score theo từng cột:

$$
\tilde{\mathbf{b}}_v = \frac{\log(1 + \mathbf{b}_v) - \boldsymbol{\mu}_{tr}}{\boldsymbol{\sigma}_{tr}}, \tag{3}
$$

trong đó $\boldsymbol{\mu}_{tr}, \boldsymbol{\sigma}_{tr}$ chỉ được ước lượng trên
$\mathcal{V}_{tr}$. Cột nào có $\sigma = 0$ thì đặt $\sigma = 1$.

**Ngữ cảnh học viên.** $\mathbf{u}_v$ gồm mã hóa one-hot của giới tính và trình độ học
vấn, cộng với tuổi tại thời điểm khóa học bắt đầu (năm bắt đầu khóa trừ năm sinh).
Mỗi biến phân loại có thêm hai hạng mục *thiếu* và *ngoài từ vựng*. Tuổi ngoài khoảng
$[10, 100]$ được xem là thiếu. Giá trị thiếu được thay bằng trung vị của
$\mathcal{V}_{tr}$, sau đó chuẩn hóa z-score theo thống kê của $\mathcal{V}_{tr}$ và đi
kèm một cờ nhị phân đánh dấu thiếu.

**Ngữ cảnh khóa học.** $\mathbf{c}_v$ là mã hóa one-hot của lĩnh vực khóa học (17 lĩnh
vực, cộng hai hạng mục *thiếu* và *ngoài từ vựng*).

Mọi từ vựng được cố định từ trước, còn mọi thống kê chuẩn hóa chỉ lấy từ
$\mathcal{V}_{tr}$. Nhờ vậy đặc trưng của tập kiểm định và tập kiểm tra không ảnh hưởng
đến phép biến đổi áp dụng lên chính chúng.

---

## 3.3. Xây dựng siêu đồ thị ban đầu

Siêu đồ thị huấn luyện là $\mathcal{G}_0 = (\mathcal{V}_{tr}, \mathcal{E}, \mathbf{H}_0)$
với $N = |\mathcal{V}_{tr}|$ nút. $H_0(v, e) = 1$ khi và chỉ khi $v \in e$. Tập siêu cạnh
là hợp của bốn họ

$$
\mathcal{E} = \mathcal{E}_{\text{course}} \cup \mathcal{E}_{\text{object}} \cup \mathcal{E}_{\text{beh}} \cup \mathcal{E}_{\text{self}}. \tag{4}
$$

**Siêu cạnh khóa học.** Với mỗi khóa $c$, siêu cạnh
$e_c = \{ v \in \mathcal{V}_{tr} : \text{course}(v) = c \}$ nối mọi học viên cùng lớp.
Loại siêu cạnh này mô hình hóa ảnh hưởng chung của khóa học: độ khó, lịch học, chất
lượng bài giảng.

**Siêu cạnh học liệu.** Với mỗi học liệu $o$ thuộc loại
$\tau \in \{\text{video}, \text{problem}, \text{forum}\}$ trong khóa $c$, siêu cạnh
$e_{(c,\tau,o)}$ gồm mọi lượt ghi danh đã tương tác với $o$ trong cửa sổ quan sát.
Khóa của siêu cạnh luôn chứa định danh khóa học $c$. Lý do là cùng một học liệu thường
được dùng lại qua nhiều lần mở lớp: trên dữ liệu XuetangX có 4.424 định danh học liệu
xuất hiện ở từ hai lớp trở lên. Nếu không phân biệt $c$, học viên ở các học kỳ khác nhau
sẽ bị gộp vào cùng một quan hệ. Các hành động thuộc nhóm *web page* không có định danh
học liệu nên không tạo siêu cạnh loại này.

**Siêu cạnh hành vi.** Độ tương đồng hành vi giữa hai lượt ghi danh được đo bằng cosine
trên 58 chiều hành vi $\tilde{\mathbf{b}}_v = [\tilde{\mathbf{b}}^{\text{day}}_v \| \tilde{\mathbf{b}}^{\text{act}}_v]$.
Gọi $\mathcal{N}_K(v)$ là $K$ lượt ghi danh huấn luyện gần $v$ nhất (không tính chính
$v$). Mỗi nút $v$ là tâm (anchor) của một siêu cạnh hành vi

$$
e^{\text{beh}}_v = \{v\} \cup \mathcal{N}_k(v), \qquad k = 10. \tag{5}
$$

Các láng giềng thứ $k+1$ đến $k_{\max}$ tạo thành tập ứng viên
$\mathcal{C}(v) = \mathcal{N}_{k_{\max}}(v) \setminus \mathcal{N}_k(v)$ với $k_{\max} = 20$.
Tập này được dùng cho bước bổ sung liên thuộc tiềm ẩn (Mục 3.5).

**Siêu cạnh tự thân.** Mỗi nút có một siêu cạnh $\{v\}$. Siêu cạnh này không bao giờ bị
loại bỏ, nhờ đó không nút nào bị cô lập sau khi học cấu trúc.

Siêu cạnh có ít hơn hai thành viên (trừ siêu cạnh tự thân) bị loại. Bảng 2 thống kê
siêu đồ thị thu được.

**Bảng 2. Thống kê siêu đồ thị huấn luyện $\mathcal{G}_0$ (XuetangX, $N = 126{.}354$).**

| Họ siêu cạnh | Số siêu cạnh | Kích thước nhỏ nhất / trung vị / lớn nhất |
|---|---:|---|
| Khóa học | 247 | 151 / 377 / 2.361 |
| Học liệu | 21.747 | 2 / 27 / 1.893 |
| Hành vi | 126.354 | 11 / 11 / 11 |
| Tự thân | 126.354 | 1 |
| **Tổng liên thuộc** | | 3.201.735 |

---

## 3.4. Bộ mã hóa siêu đồ thị

Chúng tôi dùng phép tích chập siêu đồ thị của HGNN [Feng et al., 2019], mở rộng để
nhận thêm một trọng số $w_{ve} \in \{0,1\}$ cho mỗi liên thuộc. Gọi
$\mathbf{H}_w = \mathbf{H} \odot \mathbf{W}$ là ma trận liên thuộc có trọng số. Bậc
của nút và bậc của siêu cạnh được tính là

$$
d(v) = \sum_{e} H_w(v,e), \qquad \delta(e) = \sum_{v} H_w(v,e). \tag{6}
$$

Toán tử lan truyền được định nghĩa là

$$
\mathcal{P}(\mathbf{H}_w) = \mathbf{D}_v^{-1/2} \mathbf{H}_w \mathbf{D}_e^{-1} \mathbf{H}_w^{\top} \mathbf{D}_v^{-1/2}, \tag{7}
$$

với $\mathbf{D}_v = \text{diag}(\max(d(v), 1))$ và $\mathbf{D}_e = \text{diag}(\max(\delta(e), 1))$.
Bậc được tính với gradient bị chặn (stop-gradient). Nhờ đó một siêu cạnh bị loại hết
thành viên chỉ truyền giá trị 0 và không gây chia cho 0.

Bộ mã hóa hai lớp $f_\theta$ được viết là

$$
\mathbf{Z} = f_\theta(\mathbf{X}, \mathbf{H}_w) = \text{ReLU}\Big( \mathcal{P}(\mathbf{H}_w)\, \text{Dropout}\big(\text{ReLU}(\mathcal{P}(\mathbf{H}_w)\,(\mathbf{X}\boldsymbol{\Theta}_1 + \mathbf{b}_1))\big)\, \boldsymbol{\Theta}_2 + \mathbf{b}_2 \Big). \tag{8}
$$

Nói cách khác, mỗi lớp lan truyền theo hai bước: nút → siêu cạnh → nút.

---

## 3.5. Học cấu trúc siêu đồ thị

Siêu đồ thị $\mathbf{H}_0$ được xây dựng theo quy tắc cố định nên có thể vừa chứa liên
kết nhiễu (ví dụ những học viên cùng khóa nhưng hành vi rất khác nhau), vừa thiếu liên
kết hữu ích (ví dụ láng giềng hành vi nằm ngay ngoài ngưỡng $k$). Theo HSL
[Cai et al., 2022], chúng tôi học một cấu trúc tinh chỉnh

$$
\mathbf{H}^* = \mathbf{M}_e \odot \mathbf{M}_v \odot (\mathbf{H}_0 + \Delta\mathbf{H}) + \mathbf{I}. \tag{9}
$$

Mọi thành phần của (9) được tính từ biểu diễn ban đầu $\mathbf{Z}^0 = f_\theta(\mathbf{X}, \mathbf{H}_0)$.
Biểu diễn của một siêu cạnh là trung bình biểu diễn các thành viên của nó trong $\mathbf{H}_0$:

$$
\mathbf{h}_e = \frac{1}{|e|} \sum_{v \in e} \mathbf{z}^0_v. \tag{10}
$$

**Bổ sung liên thuộc tiềm ẩn $\Delta\mathbf{H}$.** Với mỗi siêu cạnh hành vi
$e = e^{\text{beh}}_a$, mô hình chấm điểm các ứng viên $u \in \mathcal{C}(a)$ theo

$$
s(u, e) = \cos\big(\mathbf{z}^0_u, \mathbf{h}_e\big), \tag{11}
$$

và thêm $r = 2$ ứng viên có điểm cao nhất vào $e$. Bước chọn top-$r$ không khả vi và
không truyền gradient. Các liên thuộc mới được giữ hay loại là do mặt nạ $\mathbf{M}_v$
quyết định ở bước sau. $\Delta\mathbf{H}$ chỉ được áp dụng cho họ siêu cạnh hành vi, vì
thêm một học viên khóa khác vào siêu cạnh khóa học hay siêu cạnh học liệu sẽ làm sai ý
nghĩa của các quan hệ đó.

**Lấy mẫu siêu cạnh $\mathbf{M}_e$.** Xác suất giữ lại siêu cạnh $e$ phụ thuộc vào biểu
diễn của nó và họ của nó:

$$
\pi_e = \sigma\big(\text{MLP}_e([\mathbf{h}_e \,\|\, \text{onehot}(\text{family}(e))])\big). \tag{12}
$$

**Lấy mẫu liên thuộc $\mathbf{M}_v$.** Xác suất giữ lại liên thuộc $(v, e)$, áp dụng cho
cả liên thuộc gốc và liên thuộc trong $\Delta\mathbf{H}$:

$$
\pi_{ve} = \sigma\big(\text{MLP}_v([\mathbf{z}^0_v \,\|\, \mathbf{h}_e])\big). \tag{13}
$$

**Lấy mẫu khả vi.** Trong huấn luyện, mỗi mặt nạ được lấy mẫu từ phân phối Bernoulli nới
lỏng với nhiễu logistic $g = \log U - \log(1 - U)$, $U \sim \mathcal{U}(0,1)$:

$$
\tilde{m} = \sigma\!\left(\frac{\text{logit}(\pi) + g}{\tau_g}\right), \qquad
m = \mathbb{1}[\tilde{m} > 0.5] + \tilde{m} - \text{sg}(\tilde{m}), \tag{14}
$$

với $\tau_g = 0{.}4$ và $\text{sg}(\cdot)$ là phép chặn gradient. Ước lượng
straight-through này cho giá trị nhị phân ở lượt tiến và gradient trơn ở lượt lùi. Khi
suy luận, mặt nạ là tất định: $m = \mathbb{1}[\pi > 0.5]$. Liên thuộc của siêu cạnh tự
thân luôn có $m = 1$. Hệ số chặn (bias) của lớp cuối trong hai MLP được khởi tạo bằng
3, tương ứng xác suất giữ ban đầu $\sigma(3) \approx 0{.}95$. Như vậy mô hình khởi đầu
gần với $\mathbf{H}_0$ rồi mới học dần cách loại bỏ.

Trọng số liên thuộc cuối cùng là $w_{ve} = m_e \cdot m_{ve}$. Biểu diễn tinh chỉnh và
xác suất bỏ học được tính bằng *cùng* bộ mã hóa:

$$
\mathbf{Z}^* = f_\theta(\mathbf{X}, \mathbf{H}^*), \qquad \hat{y}_v = \sigma(\mathbf{w}_c^{\top}\mathbf{z}^*_v + b_c). \tag{15}
$$

**Nhận xét.** $\mathbf{H}^*$ được sinh lại từ $\mathbf{H}_0$ ở mỗi vòng lặp huấn luyện,
không tích lũy từ vòng trước. Thứ mà mô hình học được là *quy tắc chấm điểm*
$(\text{MLP}_e, \text{MLP}_v)$, không phải một siêu đồ thị cố định. So với HSL gốc,
chúng tôi có hai điều chỉnh để phù hợp với bài toán quy nạp:
(i) $\mathbf{M}_e$ được tham số hóa bằng MLP trên $\mathbf{h}_e$ thay vì một tham số tự
do cho mỗi siêu cạnh, nhờ đó áp dụng được lên các siêu cạnh chưa từng gặp trong đồ thị
cục bộ khi suy luận (Mục 3.8);
(ii) $\Delta\mathbf{H}$ chỉ bổ sung vào siêu cạnh hành vi.

---

## 3.6. Học tương phản trong siêu cạnh

Cặp $\mathbf{H}_0$ và $\mathbf{H}^*$ tự nhiên cho ta hai góc nhìn của cùng một lượt ghi
danh. Trong học tương phản trên dữ liệu chuỗi, góc nhìn thứ hai thường được tạo bằng
tăng cường dữ liệu đầu vào (che bước thời gian, che đặc trưng). Ở đây, góc nhìn thứ hai
được tạo bằng **tăng cường cấu trúc có học**: $\mathbf{M}_e$ loại một số quan hệ,
$\mathbf{M}_v$ loại một số thành viên, $\Delta\mathbf{H}$ bổ sung láng giềng mới. Thêm
vào đó, hai lượt mã hóa dùng mặt nạ dropout độc lập nhau, tương đương một dạng che đặc
trưng trên không gian ẩn.

Hàm mất mát tương phản có hai mục đích:
(i) **nhất quán**: biểu diễn của cùng một nút trên hai cấu trúc phải gần nhau, qua đó
ràng buộc $\mathbf{H}^*$ không rời quá xa $\mathbf{H}_0$;
(ii) **phân biệt**: các nút cùng siêu cạnh phải phân biệt được với nhau, nhằm chống hiện
tượng làm mượt quá mức (over-smoothing) do phép lấy trung bình trong các siêu cạnh lớn.

**Cặp dương và tập âm.** Chuẩn hóa $\bar{\mathbf{z}} = \mathbf{z}/\|\mathbf{z}\|_2$. Với
mỗi nút neo $i$, cặp dương là $(\bar{\mathbf{z}}^0_i, \bar{\mathbf{z}}^*_i)$. Tập âm
$\mathcal{T}_i$ gồm $K$ nút lấy mẫu có hoàn lại từ các siêu cạnh chứa $i$ trong
$\mathbf{H}_0$ (không tính siêu cạnh tự thân). Mỗi lần lấy mẫu, trước hết chọn đều một
siêu cạnh $e \ni i$, sau đó chọn đều một thành viên $j \in e$. Nếu $j = i$ thì phần tử
đó bị loại khỏi mẫu số. Vì cùng siêu cạnh với $i$, các nút âm này là mẫu âm khó (hard
negatives).

**Hàm mất mát.** Với góc nhìn neo $\mathbf{A}$ và góc nhìn còn lại $\mathbf{B}$:

$$
\ell(i; \mathbf{A}, \mathbf{B}) = -\log
\frac{\exp(\bar{\mathbf{a}}_i^{\top}\bar{\mathbf{b}}_i / \tau)}
{\exp(\bar{\mathbf{a}}_i^{\top}\bar{\mathbf{b}}_i / \tau)
+ \sum_{j \in \mathcal{T}_i} \Big[ \exp(\bar{\mathbf{a}}_i^{\top}\bar{\mathbf{a}}_j / \tau) + \exp(\bar{\mathbf{a}}_i^{\top}\bar{\mathbf{b}}_j / \tau) \Big]}. \tag{16}
$$

Hàm mất mát được lấy đối xứng theo hai chiều và trung bình trên một tập nút neo
$\mathcal{B}$ lấy mẫu ngẫu nhiên ở mỗi vòng lặp:

$$
\mathcal{L}_{\text{CL}} = \frac{1}{2|\mathcal{B}|} \sum_{i \in \mathcal{B}}
\Big[ \ell(i; \mathbf{Z}^0, \mathbf{Z}^*) + \ell(i; \mathbf{Z}^*, \mathbf{Z}^0) \Big]. \tag{17}
$$

Chúng tôi dùng $|\mathcal{B}| = 1024$, $K = 32$ và $\tau = 0{.}07$. Gradient của
$\mathcal{L}_{\text{CL}}$ đi qua $\mathbf{Z}^0$ và $\mathbf{Z}^*$ vào bộ mã hóa
$f_\theta$, đồng thời đi qua các trọng số $w_{ve}$ (nhờ ước lượng straight-through ở
(14)) vào $\text{MLP}_e$ và $\text{MLP}_v$. Vì vậy học tương phản cũng đóng vai trò chính
quy hóa cho mô-đun học cấu trúc. Hàm mất mát này chỉ dùng khi huấn luyện.

---

## 3.7. Hàm mục tiêu và huấn luyện

Tỉ lệ bỏ học trong dữ liệu khoảng 76%. Để cân bằng hai lớp, chúng tôi dùng entropy chéo
nhị phân có trọng số, với trọng số lớp dương $\omega = N_{\text{neg}} / N_{\text{pos}}$
tính trên $\mathcal{V}_{tr}$:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{v \in \mathcal{V}_{tr}} \Big[ \omega\, y_v \log \hat{y}_v + (1 - y_v) \log (1 - \hat{y}_v) \Big]. \tag{18}
$$

Hàm mục tiêu tổng là

$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda\, \mathcal{L}_{\text{CL}}, \qquad \lambda = 0{.}1. \tag{19}
$$

Mô hình được huấn luyện toàn bộ (full-batch) trên $\mathcal{G}_0$: mỗi epoch gồm một
lượt tiến và một bước cập nhật Adam. Sau mỗi 5 epoch, mô hình được đánh giá trên
$\mathcal{V}_{val}$. Điểm lưu (checkpoint) có AUC kiểm định cao nhất được giữ lại, và
quá trình huấn luyện dừng sớm khi AUC kiểm định không cải thiện sau 5 lần đánh giá liên
tiếp. Tập kiểm tra chỉ được đánh giá **một lần** trên điểm lưu đã chọn, với ngưỡng phân
loại cố định 0,5.

**Thuật toán 1. Một epoch huấn luyện.**

```
Đầu vào: X, H0, tập ứng viên C(·), tham số θ, φ = (MLP_e, MLP_v)
 1. Z0  ← f_θ(X, H0)                                   # góc nhìn 1
 2. h_e ← trung bình của Z0 trên các thành viên của e  # (10)
 3. ΔH  ← top-r ứng viên theo cos(z0_u, h_e)           # (11), không gradient
 4. M_e, M_v ← lấy mẫu Gumbel straight-through         # (12)–(14)
 5. H*  ← M_e ⊙ M_v ⊙ (H0 + ΔH) + I                    # (9)
 6. Z*  ← f_θ(X, H*)                                   # góc nhìn 2, cùng θ
 7. ŷ   ← σ(w_c^T Z* + b_c)                            # (15)
 8. B   ← 1024 nút neo; T_i ← 32 nút cùng siêu cạnh    # Mục 3.6
 9. L   ← L_BCE(ŷ, y) + λ · L_CL(Z0, Z*, B, T)         # (16)–(19)
10. Cập nhật θ, φ, w_c bằng Adam (cắt chuẩn gradient ở 5)
```

---

## 3.8. Suy luận quy nạp không rò rỉ

Nhiều mô hình GNN cho dự báo bỏ học làm việc theo kiểu *truyền dẫn* (transductive):
đưa cả nút kiểm tra vào đồ thị khi huấn luyện. Khi đó biểu diễn của nút huấn luyện nhận
thông điệp từ đặc trưng của nút kiểm tra, và các nút kiểm tra ảnh hưởng lẫn nhau qua các
siêu cạnh chung. Để loại bỏ rò rỉ này, chúng tôi dùng giao thức *quy nạp* (inductive)
như sau.

**Siêu đồ thị cục bộ.** Với mỗi nút đích $t \in \mathcal{V}_{val} \cup \mathcal{V}_{te}$,
chúng tôi dựng một siêu đồ thị cục bộ $\mathcal{G}_t$ gồm $t$ và một tập các nút huấn
luyện $\mathcal{U}_t \subset \mathcal{V}_{tr}$. Các siêu cạnh của $\mathcal{G}_t$ là:

- $\{t\} \cup e_c$: siêu cạnh khóa học của $t$, chỉ gồm thành viên huấn luyện;
- $\{t\} \cup e_{(c,\tau,o)}$: với mỗi học liệu $o$ mà $t$ đã dùng và có siêu cạnh tương ứng trong $\mathcal{G}_0$;
- $\{t\} \cup \mathcal{N}_k(t)$: siêu cạnh hành vi của $t$, với láng giềng tìm trong $\mathcal{V}_{tr}$ theo (5);
- siêu cạnh tự thân cho mọi nút của $\mathcal{G}_t$.

Tập ứng viên $\mathcal{C}(t)$ (láng giềng thứ $k+1$ đến $k_{\max}$ trong
$\mathcal{V}_{tr}$) được dùng cho $\Delta\mathbf{H}$ trên siêu cạnh hành vi của $t$.
$\mathcal{U}_t$ là hợp của mọi thành viên huấn luyện kể trên. Đặc trưng của $t$ được biến
đổi bằng thống kê của $\mathcal{V}_{tr}$ (Mục 3.2).

**Dự báo.** Mô hình đã huấn luyện được áp dụng lên $\mathcal{G}_t$ với mặt nạ tất định
và không có dropout. Chỉ đầu ra tại nút $t$ được dùng làm $\hat{y}_t$. Để tăng tốc, nhiều
siêu đồ thị cục bộ được ghép thành một siêu đồ thị khối chéo. Vì không có siêu cạnh nào
nối hai siêu đồ thị cục bộ, các nút đích không nhìn thấy nhau.

**Tính chất.** Giao thức trên bảo đảm bốn điều:
(i) không nút kiểm định hay kiểm tra nào xuất hiện trong $\mathcal{G}_0$ hoặc tham gia
lượt tiến khi huấn luyện;
(ii) mọi phép biến đổi đặc trưng và mọi tìm kiếm láng giềng chỉ dựa trên
$\mathcal{V}_{tr}$;
(iii) nhãn không phải là đầu vào của mô hình, và nhãn của nút đích chỉ dùng để tính độ đo;
(iv) mọi lựa chọn mô hình dựa trên $\mathcal{V}_{val}$.

Cái giá của giao thức này là cấu trúc lúc suy luận khác với lúc huấn luyện. Nút đích chỉ
thuộc một siêu cạnh hành vi, và các nút huấn luyện trong $\mathcal{G}_t$ không mang theo
lân cận riêng của chúng trong $\mathcal{G}_0$.

---

## 3.9. Siêu tham số

**Bảng 3. Siêu tham số mặc định.**

| Nhóm | Siêu tham số | Giá trị |
|---|---|---|
| Dữ liệu | Cửa sổ quan sát $T$ | 35 ngày |
| | Tỉ lệ train : validation | 80 : 20 (seed 1) |
| Siêu đồ thị | $k$ / $k_{\max}$ | 10 / 20 |
| Bộ mã hóa | Số lớp / chiều ẩn / dropout | 2 / 128 / 0,3 |
| HSL | Số liên thuộc bổ sung $r$ mỗi siêu cạnh hành vi | 2 |
| | Nhiệt độ Gumbel $\tau_g$ | 0,4 |
| | Chiều ẩn của MLP chấm điểm | 32 |
| | Xác suất giữ ban đầu | $\sigma(3) \approx 0{,}95$ |
| Tương phản | $\lambda$ / $\tau$ | 0,1 / 0,07 |
| | Số nút neo $\lvert\mathcal{B}\rvert$ / số mẫu âm $K$ | 1024 / 32 |
| Tối ưu | Adam: learning rate / weight decay | $10^{-3}$ / $5 \times 10^{-4}$ |
| | Cắt chuẩn gradient | 5 |
| | Số epoch tối đa / tần suất đánh giá / patience | 200 / 5 / 5 |
| | Seed | 1, 11, 111, 1111, 11111 |

---

### Tài liệu tham khảo

- Y. Feng, H. You, Z. Zhang, R. Ji, Y. Gao. *Hypergraph Neural Networks*. AAAI 2019, pp. 3558–3565.
- D. Cai, M. Song, C. Sun, B. Zhang, S. Hong, H. Li. *Hypergraph Structure Learning for Hypergraph Neural Networks*. IJCAI 2022, pp. 1923–1929.
