# Hướng dẫn đọc code

Nếu cần giải thích theo từng khối trong sơ đồ, input/output và chi tiết tám file,
xem [tài liệu project đầy đủ](project-guide.md).

Đọc tám file trong `src/` theo số thứ tự dưới đây. Mỗi file có một hàm chính,
các hàm còn lại chỉ giúp thực hiện bước đó.

```text
1 download.py     download()          → sáu CSV gốc
2 preprocess.py   preprocess()        → nodes, events 35 ngày, split theo user
3 features.py     build_features()    → X cho từng seed
4 hypergraph.py   build_hyperedges()  → danh sách edge và membership
5 model.py        build_h0()          → sparse H0, train node order
                  HGSLModel.forward() → Z0, H*, Z*, logits
6 hsl.py          refine_hypergraph() → sample và score membership
7 losses.py       total_loss()        → weighted BCE + λ InfoNCE
8 train.py        train(), test()     → checkpoint theo validation, rồi test
```

## Theo dấu một node

Một dòng `train_truth.csv` hoặc `test_truth.csv` có `enroll_id` và `truth`.
`preprocess.py` ghép nó với log cùng `enroll_id`, gán `node_id` liên tiếp, và ghi
`nodes.csv`. `source_partition` chỉ cho biết file gốc; `split_seed_*.csv` mới là
train/validation/test của thí nghiệm. Mọi enrollment của cùng một user nằm cùng
một split.

`features.py` duyệt event ngày 0–34 để tạo 35 day counts, 23 action counts,
session count và số object khác nhau: 60 chiều. User/course context thêm 36 chiều.
Mean, standard deviation và median chỉ tính trên node train của từng seed.
`X_seed_*.npy` có thứ tự hàng đúng bằng `node_id`.

`hypergraph.py` nối các train node chung course, chung `(course_id, object_id,
object_type)`, hoặc gần nhau theo cosine của 60 behavioral feature. `model.py`
chuyển danh sách membership thành ma trận thưa `H0` với hàng theo
`train_ids_seed_*.npy`. Khi đánh giá, `local_graph()` dựng một graph cho target
validation/test cùng các train reference; target khác không đi vào graph này.

## Theo dấu một forward

```text
X + H0 → HGNN → Z0
Z0 + H0 → sample edge/node → score membership → H*
X + H* → cùng HGNN → Z* → classifier → dropout logit
Z0 + Z* → contrastive loss
logit + train label → weighted BCE
total = BCE + λ × contrastive
```

Trong `HGSLModel.forward()`, `hsl.py` được gọi sau khi có `Z0`. Train chọn một
số edge mỗi epoch. Validation/test chọn toàn bộ edge trong local graph với
candidate cố định, để đổi evaluation batch size không đổi kết quả.

`train.py` chỉ dùng validation AUC để giữ checkpoint tốt nhất. Hàm `test()`
đọc checkpoint đó rồi mới tính test metrics. `--no-hsl` dùng cùng encoder nhưng
bỏ bước refine và contrastive loss, phục vụ ablation. Checkpoint ghi cấu hình và
mã kiểm tra của các file dữ liệu đã dùng; nếu em tạo lại feature, graph hoặc
split, `test()` yêu cầu train lại thay vì trả về metric từ dữ liệu không khớp.

## Định dạng file

CSV/CSV nén là dữ liệu dạng bảng dễ mở. `.npy` chứa ma trận feature. `.npz`
chứa sparse incidence `H0`; không chuyển `H0` sang ma trận dense vì nó có hơn
144 nghìn node và hơn 150 nghìn edge ở seed 1.
