# XuetangX Hypergraph Structure Learning

Project này phục vụ một mô hình duy nhất: dự đoán dropout trên XuetangX bằng
Hypergraph Neural Network kết hợp học lại cấu trúc hypergraph.

Sơ đồ và cách tổ chức code được mô tả tại
[docs/model-structure.md](docs/model-structure.md).

## Quy tắc code

- Mỗi mô hình nằm trong một folder riêng dưới `src/mooc_hgsl/models/`.
- Mỗi khối trên flowchart là một file, không gom toàn bộ model vào một file lớn.
- Không dùng `configs/`, YAML hoặc TOML. Đường dẫn ổn định nằm trong `paths.py`;
  siêu tham số được truyền qua constructor hoặc CLI.
- Một file chỉ có một trách nhiệm chính; tên class, hàm và biến phải nói rõ ý nghĩa.
- Comment ngắn, tự nhiên, giải thích lý do hoặc bẫy dữ liệu; không diễn giải lại
  từng dòng code.
- Full activity log không có `object` và `truth`; không tự tạo nhãn giả.
- Feature transformation và hypergraph dùng để train phải được fit trên train.
- Incidence matrix luôn lưu sparse, không tạo `H` dense.

Chi tiết xem [docs/development-rules.md](docs/development-rules.md).
Quy ước feature/object và bảng đối chiếu baseline nằm tại
[docs/xuetangx-feature-analysis.md](docs/xuetangx-feature-analysis.md).

## Cấu trúc chính

```text
src/mooc_hgsl/
  paths.py
  cli.py
  data/
    download.py
    schema.py
    preprocessing.py
    reporting.py
    pipeline.py
    source_check.py
    artifacts.py
    feature_engineering.py
  models/
    hypergraph_structure_learning/
      hypergraph_construction.py
      hgnn.py
      hyperedge_sampling.py
      incident_node_sampling.py
      hypergraph_refinement.py
      contrastive_loss.py
      classifier.py
      total_loss.py
      model.py
  training/
    trainer.py
```

`baseline/` chỉ là tài liệu/code tham khảo của tác giả, không thuộc package đang
phát triển và không được import vào mô hình đề xuất.

## Dữ liệu

- Full JSON: `data/raw/xuetangx_full/` — hai archive và sáu JSON đã giải nén.
- Full CSV: `data/processed/xuetangx_full/` — sáu CSV.GZ dùng cho pretraining.
- Prediction cohort: `data/raw/xuetangx/` — cần cho object hyperedge, nhãn dropout
  và supervised fine-tuning.
- Script dữ liệu đặt trong `scripts/`, không đặt lẫn trong `src/`.

## Lệnh kiểm tra

```powershell
.\.venv\Scripts\python.exe run.py structure
.\.venv\Scripts\python.exe run.py check-data
.\.venv\Scripts\python.exe run.py process-data
.\.venv\Scripts\python.exe run.py build-feature-store
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`check-data` trả mã lỗi nếu thiếu prediction cohort. Tải lại bằng:

```powershell
.\.venv\Scripts\python.exe scripts/download_xuetangx_prediction.py
```

Full activity log dùng đúng hai script, không đi qua package/model code:

```powershell
.\.venv\Scripts\python.exe scripts/download_xuetangx_full.py
.\.venv\Scripts\python.exe scripts/convert_xuetangx_full.py
```

`process-data` quét toàn bộ hai nhánh XuetangX, tạo artifact Parquet trong
`data/processed/xuetangx_feature_store/v1/` và báo cáo chi tiết trong
`outputs/reports/xuetangx_hgsl/`. Project không dùng file config ngoài.

`build-feature-store` chỉ đọc Parquet đã xử lý để tạo object catalog và các view
dùng chung cho SIG-Net, MST-GCN và mô hình đề xuất. Nếu manifest nguồn không đổi,
lệnh trả về cache ngay và không quét lại 40 triệu event.

## Trạng thái

Phase 1 đã hoàn tất: tải/kiểm tra nguồn, tạo enrollment-node, lọc cửa sổ 35 ngày,
tạo feature thô và xuất báo cáo. Kế hoạch cho HGNN, structure learning, loss và
đánh giá nằm tại [docs/implementation-plan.md](docs/implementation-plan.md).
