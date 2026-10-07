# Rà soát cách validation / test và phương án thay thế

Ngày 2026-10-07. File này chỉ là kế hoạch, chưa sửa code.

## 1. Cách hiện tại

```
Train:      H0 = chỉ các nút train  ──► học full-batch, loss trên nút train
Val / Test: từng target t được chèn RIÊNG vào H0
            - t nhập các hyperedge của mình (δ(e) + 1)
            - các nút train giữ trạng thái đã tính trên H0 (không cập nhật)
            - t không thấy các target khác (val/test không nối với nhau)
```

Code: `load_targets` (5_graph_data), `cache_train_states` + `forward_targets` (6_hgnn), `predict` (9_train).

Vấn đề:
- Không có công trình HGNN nào chấm theo kiểu "chèn từng nút một, các nút test cô lập với nhau" (xem mục 2). Reviewer sẽ hỏi và em phải tự bảo vệ một giao thức không có tiền lệ.
- Nó cần 2 hàm riêng chỉ để chấm điểm, khó đọc, khó kiểm.

## 2. Các công trình làm thế nào

| Công trình | Đồ thị lúc train | Nút test | Chọn model |
|---|---|---|---|
| HGNN (Feng et al., AAAI 2019) — code gốc | H dựng trên **mọi** đối tượng train + test (ModelNet40, NTU2012) | Có mặt trong H từ đầu (transductive) | Pha `val` dùng `idx_test`, giữ model có acc test cao nhất — **chọn trên test, không nên chép** |
| HGNN+ / thư viện DHG (Gao et al., TPAMI 2022) | Hypergraph trên mọi đỉnh, có `train_mask`, `val_mask`, `test_mask` | Có mặt từ đầu; chạy forward toàn đồ thị rồi lọc theo mask | Best theo val, chấm test một lần |
| AllSet (Chien et al., ICLR 2022) | Toàn hypergraph | Có mặt từ đầu | Chia ngẫu nhiên 50/25/25, 20 lần; lấy test tại epoch val tốt nhất |
| ED-HNN (Wang et al., ICLR 2023) | Theo AllSet | Có mặt từ đầu | 50/25/25, 10 lần |
| LightHGNN (Feng, …, Gao, ICLR 2024) — **cùng nhóm tác giả HGNN/HGNN+** | (a) Transductive: G lớn chứa train + val + test. (b) "Production": bỏ 20% test (V_ind) ra khỏi G khi train | (b) Lúc suy luận dùng G lớn: **mọi** nút V_ind vào đồ thị **cùng lúc** | Best theo val; báo cáo transductive / inductive / production |
| HyperSAGE (Arya et al., 2020) | Inductive: nút unseen không có trong hypergraph lúc train | Lúc test dùng toàn bộ láng giềng | — |
| GraphSAGE (Hamilton et al., NeurIPS 2017) | Chỉ dữ liệu train (chia theo thời gian: Reddit, citation) | Nút mới được nhúng bằng hàm tổng hợp đã học | — |
| SIG-Net, MST-GCN, CA-TFHN (baseline MOOC, đã audit 2026-10-04) | Đồ thị trên train + test | Có mặt từ đầu (transductive) | Không có val; MST-GCN/CA-TFHN lấy epoch cuối hoặc in test mỗi epoch |

Kết luận:
1. Giao thức **chủ đạo** của HGNN, HGNN+, AllSet, ED-HNN và cả các baseline MOOC là **transductive**: dựng hypergraph trên mọi nút, chỉ dùng nhãn train trong loss, chọn model theo val, chấm test một lần.
2. Khi muốn đánh giá nút chưa thấy (inductive), các bài có tiền lệ (LightHGNN, HyperSAGE, GraphSAGE) đưa **cả tập** nút mới vào đồ thị cùng lúc, không chèn từng nút một.
3. Không ai dùng cách hiện tại của mình.

## 3. Phương án đề xuất

### PA1 (chính, thầy khuyến nghị): transductive như HGNN+ / DHG

```
H = Course + Object + User + self-loop trên MỌI enrollment (train + val + test)
X = cả 3 split (scaler vẫn fit trên train, như hiện nay)

mỗi epoch:  Z = model(X, H)                  # một forward toàn đồ thị
            loss = BCE(Z[train], y[train])    # chỉ nhãn train
mỗi 5 epoch: AUC(Z[val], y[val])  -> early stopping (giữ nguyên patience 40)
cuối:        nạp best state, một forward, chấm Z[val], Z[test] tại ngưỡng 0.5
```

Vì sao hợp lệ:
- Nhãn val/test không bao giờ vào loss; chỉ đặc trưng và cấu trúc của chúng được dùng — đúng định nghĩa transductive trong LightHGNN (phụ lục D) và DHG.
- Hợp với dữ liệu: test chính thức của XuetangX là chia ngẫu nhiên theo enrollment, cùng khoảng thời gian, đủ 247 khóa. Log ngày 0–34 của mọi enrollment đang học đều có sẵn tại lúc dự đoán.
- So sánh công bằng hơn với MST-GCN, CA-TFHN, SIG-Net (đều transductive).

Sửa code (gọn hơn hiện tại):
| File | Thay đổi |
|---|---|
| 4_hypergraph.py | Đọc train + validation + test, đánh số nút liên tiếp, dựng H trên tất cả; lưu `labels` và `split` (0/1/2) thay cho `train_labels`, `train_users` |
| 5_graph_data.py | Bỏ `load_targets`; một hàm load trả X (ghép 3 split), labels, masks; `apply_scenario` chỉ lọc họ hyperedge và cột |
| 6_hgnn.py | Bỏ `cache_train_states`, `forward_targets` |
| 8_model.py | Bỏ nhánh gọi `forward_targets` |
| 9_train.py | `predict` = một forward ở chế độ eval rồi lấy theo mask |

Ước lượng bớt khoảng 150 dòng. Mỗi epoch nặng hơn vì thêm nút val + test (vẫn full-batch).

Hệ quả: mọi số đã có (M = 0.8759 và các ablation) không còn so được; phải dựng lại dữ liệu từ bước 4 và chạy lại `--scenario all` × 5 seeds.

### PA2 (bổ sung, tùy chọn): production setting như LightHGNN

Chỉ để trả lời câu hỏi "nút mới thì sao?", chạy riêng cho M (5 seeds):
- Lấy ngẫu nhiên 20% test (seed cố định) làm V_ind.
- Train trên G_sub = H bỏ V_ind (và các membership của chúng); chọn model theo val như PA1.
- Chấm: transductive trên 80% test còn lại (G_sub); inductive trên V_ind với G đầy đủ, mọi V_ind vào cùng lúc; production = toàn test với G đầy đủ.
- Code thêm: một mask V_ind, một tham số `--setting`; không cần lại `forward_targets`.

### Không khuyến nghị: giữ cách hiện tại

Không có tiền lệ, các nút test bị cô lập với nhau và nút train không cập nhật — một giả định riêng mà em phải tự chứng minh.

## 4. Validation: giữ nguyên

- 80/20 ngẫu nhiên từ train chính thức (SPLIT_SEED 1), khớp cách tạo test chính thức (audit 2026-10-04). Giống DHG/AllSet dùng một tập val riêng.
- Early stopping theo val AUC, test chỉ chấm một lần bằng best state. Không chọn theo test (lỗi của code HGNN gốc và CA-TFHN).
- Test chính thức giữ cố định để so với baseline; 5 seed đổi khởi tạo model.

## 5. Em cần quyết định

1. PA1 một mình, hay PA1 + PA2?
2. Báo thầy hướng dẫn: kết quả chính chuyển sang transductive (như HGNN+), số M sẽ thay đổi.

Câu gợi ý cho bài báo (PA1):
> Following HGNN+ and AllSet, we adopt the transductive setting: the hypergraph is built over all enrollments, only training labels supervise the loss, the model with the best validation AUC is selected, and the test set is scored once.

## Tài liệu

- HGNN code: https://github.com/iMoonLab/HGNN (`train.py`, `datasets/visual_data.py`)
- HGNN+ (TPAMI 2022): https://www.semanticscholar.org/paper/HGNN%2B:-General-Hypergraph-Neural-Networks-Gao-Feng/4d72c36065a89dc02d54b5be8078864a9685077c ; ví dụ DHG: https://deephypergraph.readthedocs.io/en/latest/examples/vertex_cls/hypergraph.html
- AllSet code: https://github.com/jianhao2016/AllSet (`src/train.py`)
- ED-HNN (ICLR 2023): https://arxiv.org/abs/2207.06680
- LightHGNN (ICLR 2024), phụ lục D: https://proceedings.iclr.cc/paper_files/paper/2024/file/7fc0aac0d700873ac520af9b05e3f2b6-Paper-Conference.pdf
- HyperSAGE: https://arxiv.org/abs/2010.04558
- GraphSAGE: https://arxiv.org/abs/1706.02216
