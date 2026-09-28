# Graph chạy thế nào khi train, validation và test

Tài liệu này mô tả một lần `python src/8_train.py --mode both --seeds 1` chạy từ
đầu đến cuối: dữ liệu nào vào model, graph trông ra sao ở từng giai đoạn, và
mỗi epoch làm gì. Số liệu là số thật của bộ dữ liệu hiện tại (`k = 10`,
`hidden_dim = 128`) và của lần chạy seed 1 (best epoch 15, dừng ở epoch 40).

Code tương ứng: [8_train.py](../src/8_train.py) (vòng lặp),
[5_model.py](../src/5_model.py) (forward),
[6_hsl.py](../src/6_hsl.py) (`H*`),
[7_losses.py](../src/7_losses.py) (loss),
[4_hypergraph.py](../src/4_hypergraph.py) (graph).

---

## 1. Bức tranh chung: có hai loại graph

Điểm dễ nhầm nhất: **train và validation/test không chạy trên cùng một graph.**

| | Train | Validation / Test |
|---|---|---|
| Graph | **một graph lớn** gồm toàn bộ train | **mỗi target một graph nhỏ** (local graph) |
| Node | 126 354 enrollment train | 1 target + các enrollment train liên quan đến nó |
| Nhãn được dùng | nhãn của cả 126 354 node, để tính loss | không dùng nhãn nào khi forward; nhãn target chỉ dùng để tính metric |
| Mỗi lần forward | cả graph, một lần (full-batch) | 8 local graph ghép lại (`eval_batch_size = 8`) |
| Tạo ở | `load_train_graph` | `build_local_graph` + `merge_local_graphs` |

Lý do: validation/test là **học viên mới** mà model chưa từng thấy (thiết lập
inductive). Nếu nhét họ vào graph train, thông tin của họ sẽ lan sang node train
khi train, gây leakage. Vì vậy mỗi target chỉ được "cắm" vào phần train mà nó có
liên quan, và các target không bao giờ nhìn thấy nhau.

```text
                   ┌──────────────────────── TRAIN GRAPH (cố định) ────────────────────────┐
                   │ 126 354 node train                                                     │
                   │ 247 course + 21 747 object + 126 354 behavioral + 126 354 self-loop    │
                   │ = 274 702 hyperedge, 3 201 735 membership                              │
                   └────────────────────────────────────────────────────────────────────────┘
                              ▲ lấy feature của các node train liên quan (không lấy nhãn)
          ┌───────────────────┼────────────────────┐
  local graph của          local graph của       ...   (31 589 cho validation, 67 699 cho test)
  validation target 0      validation target 1
  = target + 753 node train
```

---

## 2. Trước vòng lặp: nạp dữ liệu (chạy một lần)

[8_train.py:153-171](../src/8_train.py#L153-L171)

```text
train_data      = load_train_graph()            X (126354, 92), labels (126354,), graph H0 + self-loop
validation_data = load_evaluation_split("validation")   nguyên liệu để dựng local graph, chưa dựng
sampler         = build_neighbor_sampler(graph)          index để lấy mẫu láng giềng cho contrastive loss
pos_weight      = 0.319                                   #không dropout / #dropout
model           = HSLModel(92, 128, dropout=0.3, HSL bật)
optimizer       = Adam(lr = 1e-3, weight_decay = 5e-4)
```

Graph train **không đổi** trong suốt quá trình train. Thứ thay đổi mỗi epoch là
`H*`, cấu trúc mà HSL lọc ra từ graph này (mục 3).

---

## 3. Một epoch train = một bước cập nhật trên toàn bộ graph

[8_train.py:180-196](../src/8_train.py#L180-L196)

Ở đây **một epoch không phải là duyệt qua nhiều mini-batch**. Mỗi epoch là **đúng
một lần forward + backward trên toàn bộ graph train**, tức một bước cập nhật của
optimizer. 200 epoch nghĩa là tối đa 200 bước.

```text
model.train()                          dropout BẬT, mask HSL được rút NGẪU NHIÊN
│
├─ ① Z0 = HGNN(X, H0)                  (126354, 128)   lượt 1: trên graph gốc
│     X ─Linear─► lan truyền trên H0 ─ReLU─Dropout─Linear─► lan truyền ─ReLU─► Z0
│
├─ ② h_e = trung bình Z0 của thành viên  (274702, 128)  một vector cho mỗi hyperedge
│
├─ ③ HSL dựng H* = Me ⊙ Mv ⊙ (H0 + ΔH) + I
│     ΔH : mỗi Behavioral hyperedge kéo thêm 2 ứng viên gần nhất  → +252 708 membership
│     Me : giữ/bỏ từng hyperedge          (rút Gumbel, ngẫu nhiên mỗi epoch)
│     Mv : giữ/bỏ từng membership         (rút Gumbel, ngẫu nhiên mỗi epoch)
│     I  : self-loop luôn giữ
│     → weights (3 454 443,) gồm các giá trị 0/1
│
├─ ④ Z* = HGNN(X, H*)                  (126354, 128)   lượt 2: CÙNG trọng số với ①
│
├─ ⑤ logits = Linear(Z*)               (126354,)       điểm dropout của MỌI node train
│
├─ ⑥ loss = BCE(logits, 126354 nhãn; pos_weight)
│         + 0.1 · contrastive(Z0, Z*)   trên 1024 anchor ngẫu nhiên, mỗi anchor 32 láng giềng
│
└─ ⑦ loss.backward() → clip gradient (≤ 5) → optimizer.step()
      cập nhật: layer1, layer2, classifier, edge_scorer, membership_*
```

**Những gì được học** (có gradient): hai layer HGNN, classifier, MLP của `Me`, MLP
của `Mv`. **Những gì không học**: bản thân `H0` và ΔH (chọn top-k, không có
gradient).

**Vì sao `H*` khác nhau mỗi epoch:** khi train, `Me` và `Mv` được *rút ngẫu
nhiên* theo xác suất mà MLP dự đoán. Giống dropout, nhưng áp lên cấu trúc graph:
mỗi epoch model thấy một phiên bản graph hơi khác, còn MLP thì học xem hyperedge
và membership nào đáng giữ.

Log của mỗi epoch in ra đúng các thứ trên:

```text
[train] epoch 7: loss=0.71, bce=0.55, contrastive=1.62, train_auc=0.79,
        kept_course=0.95, kept_object=0.94, kept_behavioral=0.95, added=240112 (18.3s)   ← minh hoạ
```

- `kept_*`: tỉ lệ membership của `H0` còn được giữ trong `H*` ở epoch này.
- `added`: số membership ΔH còn được giữ.

---

## 4. Validation: cứ 5 epoch một lần

[8_train.py:206-227](../src/8_train.py#L206-L227),
[8_train.py:113-144](../src/8_train.py#L113-L144)

### 4.1 Dựng local graph cho một target

[4_hypergraph.py:266-305](../src/4_hypergraph.py#L266-L305)

Ví dụ thật: validation target 0 (enroll 773, khoá `TsinghuaX+70800232X+2015_T2`,
đã xem 1 video):

```text
node local 0  = target
node local 1… = các node train liên quan (753 node)

hyperedge 0  course      {target + 733 train cùng khoá}                      734 node
hyperedge 1  object      {target + 540 train đã xem cùng video}               541 node
hyperedge 2  behavioral  {target + 10 train có hành vi giống nhất}             11 node
             ứng viên ΔH: 10 train kế tiếp (hạng 11–20), chưa là thành viên
hyperedge 3… self-loop   mỗi node một cái                                     754 cái
```

Target tham gia **đúng** các hyperedge giống như khi nó được dựng trong train:
course của nó, object nó đã dùng, và Behavioral hyperedge của riêng nó (kNN chỉ
tìm trong train).

### 4.2 Chạy theo batch 8 target

```text
for mỗi nhóm 8 target:                                      (31 589 / 8 ≈ 3 949 nhóm)
    8 local graph → merge_local_graphs → một graph ghép, không có cạnh nối giữa các phần
    model.eval()                         dropout TẮT, mask HSL CỐ ĐỊNH (giữ nếu xác suất > 0,5)
    output = model(features, graph)       chạy đúng ①–⑤ như lúc train
    lấy logit của 8 node target (vị trí target_rows), bỏ qua logit của node train
sigmoid → xác suất → AUC, AUPRC, F1, precision, recall trên 31 589 target
```

Các node train trong local graph chỉ đóng vai trò **ngữ cảnh**: feature của
chúng lan truyền sang target. Logit của chúng bị bỏ đi, và nhãn của chúng không
được dùng.

### 4.3 Chọn checkpoint và early stopping

```text
nếu val AUC > AUC tốt nhất  → lưu outputs/runs/<run>.pt, đặt lại bộ đếm
ngược lại                   → bộ đếm + 1; nếu bộ đếm = patience (5) thì DỪNG
```

Lần chạy seed 1 của em:

```text
epoch:   5    10    15    20    25    30    35    40
val:     ↑     ↑     ★     ✗     ✗     ✗     ✗     ✗ → dừng
                    lưu checkpoint     5 lần không cải thiện
best_epoch = 15, epochs_run = 40
```

---

## 5. Test: chạy một lần, sau khi train xong

[8_train.py:240-255](../src/8_train.py#L240-L255)

```text
nạp outputs/runs/<run>.pt   (checkpoint tốt nhất theo validation, ví dụ epoch 15)
load_evaluation_split("test")
evaluate(...)               giống hệt mục 4.2, nhưng trên 67 699 target test
ghi outputs/reports/<run>_test.json
```

Test **không** ảnh hưởng tới việc chọn epoch hay siêu tham số, nên kết quả test
là ước lượng khách quan.

---

## 6. So sánh train mode và eval mode

| | Train (mỗi epoch) | Validation / Test |
|---|---|---|
| `model.train()` / `model.eval()` | train | eval |
| Dropout giữa 2 layer | bật (0,3) | tắt |
| Mask `Me`, `Mv` | rút ngẫu nhiên Gumbel, 0/1 | cố định: giữ nếu xác suất > 0,5 |
| ΔH | +2 ứng viên mỗi Behavioral hyperedge | giống vậy (cho Behavioral hyperedge của target) |
| Gradient | có | không (`@torch.no_grad`) |
| Loss | BCE + 0,1 · contrastive | không có loss, chỉ có metric |
| Node được chấm điểm | cả 126 354 node train | chỉ target |

---

## 7. Quy tắc chống leakage

1. Node validation/test **không bao giờ** nằm trong graph train, nên không lan
   truyền thông tin vào node train khi train.
2. kNN của target **chỉ tìm trong train**, và thống kê chuẩn hoá feature chỉ fit
   trên train (bước 3).
3. Mỗi target có graph riêng, nên **các target không thấy nhau**, kể cả khi cùng
   khoá.
4. Nhãn của node train trong local graph **không** được đưa vào model; model chỉ
   dùng feature.
5. Checkpoint được chọn bằng validation; test chỉ chạy một lần.

---

## 8. Một điểm cần biết khi diễn giải kết quả

Cùng một node train có **ngữ cảnh khác nhau** giữa lúc train và lúc đánh giá:

- **Trong graph train:** node train 0 nằm trong 53 hyperedge (1 course, nhiều
  object, Behavioral của chính nó và của các node khác chọn nó làm láng giềng).
- **Trong local graph của một target:** node đó chỉ còn những hyperedge **có chứa
  target** (tối đa: course, các object chung, Behavioral của target) cộng
  self-loop.

Vì vậy embedding của các node ngữ cảnh lúc đánh giá được tính trên một graph
thưa hơn lúc train. Target thì không bị ảnh hưởng: nó luôn có đầy đủ hyperedge
của mình. Tuy nhiên hai lớp HGNN có nghĩa là thông tin đi 2 bước (target ←
hyperedge ← node train ← hyperedge khác ← …), và bước thứ hai lúc đánh giá bị cắt
ngắn. Đây là đánh đổi của thiết lập inductive chống leakage, nên nêu trong phần
phương pháp hoặc phần hạn chế.
