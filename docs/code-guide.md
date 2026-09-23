# Hướng dẫn đọc code

Đọc các file theo thứ tự dữ liệu đi qua project:

```text
1 download.py     download()          → ba raw file
2 preprocess.py   preprocess()        → bốn bảng sạch
3 features.py     build_features()    → X theo seed
4 hypergraph.py   build_hypergraph()  → hyperedges + H0
5 model.py        HGSLModel.forward() → Z0, H*, Z*, logits
6 hsl.py          refine_hypergraph() → membership mới
7 losses.py       total_loss()        → BCE + λ InfoNCE
8 train.py        train(), test()     → checkpoint và metrics
```

## Theo dấu một enrollment

`preprocess.py` đọc truth và log trực tiếp trong `prediction_data.tar.gz`, ghép
chúng bằng `(enroll_id, source_partition)` rồi gán `node_id` liên tiếp. Log được
đọc từng dòng và event trong 35 ngày đầu được ghi ngay xuống
`events_35d.csv.gz`; không có DataFrame chứa toàn bộ log.

Split thí nghiệm được tạo xác định từ seed. Mọi enrollment của cùng một user nằm
trong cùng split. Project không ghi `split_seed_*.csv`; `train_ids_seed_*.npy`
lưu thứ tự các train node tương ứng với hàng của `H0`.

`features.py` tạo ba block:

```text
behavior: 35 day + 23 action + session count + distinct object = 60
user context:                                                   15
course context:                                                 21
total:                                                          96
```

Các event được chia vào bucket tạm để đếm session và object chính xác mà không
giữ set của hơn 40 triệu event trong RAM. Mean, standard deviation và median chỉ
được tính từ train node.

`hypergraph.py` tạo ba loại edge:

- Course: các train node cùng course.
- Object: các train node cùng `(course_id, object_id, object_type)`.
- Behavioral: anchor và `k` train neighbor gần nhất theo cosine.

Sau đó file này chuyển membership thành sparse `H0`. Khi đánh giá, mỗi target
validation/test có một local graph riêng; mọi node còn lại trong graph đều là
train reference.

## Theo dấu một forward

```text
X + H0 → HGNN → Z0
Z0 + H0 → chọn edge/node → score membership → H*
X + H* → cùng HGNN → Z* → classifier → logit
Z0 + Z* → contrastive loss
logit + train label → weighted BCE
total loss = BCE + λ × contrastive loss
```

`model.py` chỉ chứa phép truyền HGNN và `HGSLModel`. Việc dựng, lưu và nạp graph
nằm trong `hypergraph.py`. `hsl.py` giữ các bước sampling, edge embedding,
membership scoring, top-r và ghép `H*`.

`train.py` chọn checkpoint bằng validation AUC và dừng sớm khi validation không
cải thiện. Hàm `test()` chỉ đánh giá checkpoint đã được chọn; test label không
tham gia train hoặc checkpoint selection.

Để xem input/output và ý nghĩa từng artifact, đọc
[tài liệu project](project-guide.md).
