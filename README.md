# XuetangX Hypergraph Structure Learning

Project dự đoán dropout bằng Hypergraph Neural Network kết hợp học lại cấu trúc
hypergraph. Thí nghiệm chính chỉ dùng phần XuetangX có nhãn gồm **225.642
enrollment, 77.083 user và 247 course**.

## Quyết định chính

- Node là enrollment; identity chuẩn là `enroll_id`.
- Node input hỗ trợ bốn cấu hình 60/75/81/96 chiều; `full` ghép behavioral,
  user demographic và course context.
- Initial hypergraph gồm Course, Object và Behavioral hyperedge.
- User hyperedge chưa dùng trong cấu hình chính vì nguy cơ temporal leakage.
- Split chính là user-disjoint train/validation/test với tỷ lệ mục tiêu 64/16/20.
- Không dùng 351 triệu full-activity event trong thí nghiệm chính vòng đầu.
- SIG-Net và MST-GCN được chạy lại trên cùng XuetangX-247; không so trực tiếp với
  kết quả XuetangX 1.213 course đã công bố.

Chi tiết dữ liệu nằm tại
[docs/xuetangx-feature-analysis.md](docs/xuetangx-feature-analysis.md). Kế hoạch
triển khai nằm tại [docs/implementation-plan.md](docs/implementation-plan.md). Đặc tả
đối chiếu giữa sơ đồ và code nằm tại
[docs/hgsl-technical-specification.md](docs/hgsl-technical-specification.md).
Diễn giải end-to-end cho từng khối và ánh xạ tới file code nằm tại
[docs/hgsl-detailed-technical-design.md](docs/hgsl-detailed-technical-design.md).

Bản nên đọc để double-check trực tiếp từng khối trong sơ đồ với công thức, tensor,
hàm và file code tương ứng là
[docs/hgsl-model-code-walkthrough.md](docs/hgsl-model-code-walkthrough.md).

## Cấu trúc code

Code chính đặt dưới `src/`; chỉ hai nhóm xử lý nhiều bước có thư mục riêng:

```text
src/
  main.py          # pipeline end-to-end và CLI
  data.py          # chuẩn hóa raw data
  split.py         # user-disjoint train/validation/test
  features/
    engineering.py  # feature hành vi
    context.py      # user/course context
    transform.py    # train-only transform và tạo X
    io.py           # đọc X theo node_id
  hypergraph/
    structural.py   # Course và Object hyperedges
    behavioral.py   # behavioral neighbors
    audit.py        # kiểm tra hyperedges
    construction.py # tạo H0 và local memberships
  config.py        # dataset contract và cấu hình thí nghiệm
  artifacts.py     # cache, manifest và atomic write
  paths.py
  model.py         # sparse HGNN
  hsl.py           # sampling, HSL và loss
  graph_data.py    # load train/validation/test graph
  train.py         # train, validation, checkpoint và test
  metrics.py
```

`main.py` đặt hàm `run_pipeline()` ở đầu file để có thể đọc toàn bộ luồng từ raw
data đến test report mà không phải lần theo CLI hoặc các lớp trung gian.
Phần input/output của từng module được tóm tắt tại
[docs/code-guide.md](docs/code-guide.md).
Các hàm/class trong `src/` có comment `#` nêu mục đích, đầu
vào, đầu ra và lưu ý khi cần; xem
[docs/hgsl-model-code-walkthrough.md](docs/hgsl-model-code-walkthrough.md)
để đọc comment theo thứ tự các khối trong sơ đồ.

`baseline/` chứa manifest và hướng dẫn cho các repository tham khảo. Các clone cục
bộ được Git ignore và không được import vào mô hình đề xuất.

## Trạng thái

Phase 0–10 đã có luồng thực thi và kiểm thử; chưa chạy đủ thí nghiệm 5 seed/ablation.
Pipeline tạo `X_base`, `X_context`, sparse `H0`,
huấn luyện HGNN/HGSL, chọn checkpoint bằng validation và chỉ sau đó mới đánh giá
checkpoint trên test split.

Validation mặc định dùng toàn bộ split. HGSL inference refine toàn bộ local edge
theo cách deterministic, nên kết quả không đổi theo evaluation batch size.
Checkpoint được tách theo experiment ID và chỉ được load khi hash graph/feature
artifact còn khớp.

Phase 0 đã khóa:

- đường dẫn project;
- schema raw data;
- action vocabulary 23 chiều;
- dataset contract XuetangX-247;
- feature schema behavioral 60 chiều và context 36 chiều;
- năm experiment seed `1, 11, 111, 1111, 11111` và ba hyperedge family chính.

Kiểm tra contract và cấu trúc:

```powershell
.\.venv\Scripts\python.exe src/main.py contract
.\.venv\Scripts\python.exe src/main.py structure
.\.venv\Scripts\python.exe src/main.py prepare-data
.\.venv\Scripts\python.exe src/main.py split-data
.\.venv\Scripts\python.exe src/main.py build-features
.\.venv\Scripts\python.exe src/main.py build-hyperedges
.\.venv\Scripts\python.exe src/main.py build-hypergraph --behavioral-k 10
.\.venv\Scripts\python.exe src/main.py train-baseline --seed 1 --epochs 1 --feature-set full
.\.venv\Scripts\python.exe src/main.py evaluate-baseline --seed 1 --feature-set full
.\.venv\Scripts\python.exe src/main.py check-hgsl --seed 1 --feature-set full
.\.venv\Scripts\python.exe src/main.py train-hgsl --seed 1 --epochs 1 --feature-set full
.\.venv\Scripts\python.exe src/main.py evaluate --seed 1 --feature-set full
.\.venv\Scripts\python.exe src/main.py run-experiments --feature-sets behavior full
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Chạy toàn bộ pipeline từ raw data đến test trong một lệnh:

```powershell
.\.venv\Scripts\python.exe src/main.py run-pipeline --seed 1 --feature-set full --epochs 1
```

Việc còn lại của nghiên cứu là chạy đủ năm seed và các ablation mong muốn; lệnh
`run-experiments` tự ghi mean và standard deviation vào
`outputs/reports/experiment_summary.json`.
