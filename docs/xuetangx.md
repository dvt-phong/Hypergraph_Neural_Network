# Tải và kiểm tra XuetangX

Xem thêm [bảng đối chiếu tiền xử lý SIG-Net, MST-GCN và CA-TFHN](preprocessing-comparison.md), gồm các field giữ/bỏ, điều kiện lọc và khác biệt train/test.

Mở PowerShell, chạy:

```powershell
Set-Location 'D:\Project\PhD\Hypergraph_Neural_Network'
.\.venv\Scripts\python.exe scripts/download_xuetangx.py
.\.venv\Scripts\python.exe -m pip install -r scripts/requirements-data.txt
.\.venv\Scripts\python.exe scripts/profile_xuetangx.py
```

Script tự xác định thư mục gốc từ vị trí file, vì vậy không tải nhầm vào
thư mục hiện hành nếu gọi bằng đường dẫn tuyệt đối. Nếu chưa có `.venv`,
tạo bằng `python -m venv .venv` (Python 3.12 trở lên).

## Kết quả

- `data/raw/xuetangx/`: archive gốc, bốn CSV train/test và hai CSV metadata.
- `data/raw/xuetangx/download_manifest.json`: URL, dung lượng, SHA-256, thời điểm tải.
- `outputs/reports/xuetangx/statistics.md`: bảng thống kê từng field, diễn giải và top 5 giá trị.
- `outputs/reports/xuetangx/statistics.json`: cùng thống kê ở dạng máy đọc được.

Tổng tải khoảng 411 MiB. Sáu CSV giải nén chiếm 6.210.480.112 byte (5,78 GiB).
Nên có ít nhất 12 GiB trống cho dữ liệu và vùng tạm khi thống kê.
Script tải ghi file `.part`, kiểm tra dung lượng rồi mới đổi tên.
Khi chạy lại, file tải có checksum khớp manifest sẽ được tái sử dụng;
bốn CSV trong archive được giải nén lại để bảo đảm đúng bản gốc.
Checksum được tính sau khi tải, dùng kiểm tra tính nhất quán các lần chạy;
đây không phải checksum do nhà phát hành công bố.

## Nguồn và phạm vi

[MoocData](http://moocdata.cn/data/user-activity) công bố bộ
**Dropout Prediction Dataset** từ bài *Understanding Dropouts in MOOCs* (AAAI 2019).
Script tải `prediction_data.tar.gz`, `user_info.csv` và `course_info.csv`
từ máy chủ HTTPS `lfs.aminer.cn` được liên kết trên trang nguồn.
Không tải hai archive tracking log toàn nền tảng vì chúng không phải bộ train/test dự đoán.

[SIG-Net gốc](https://github.com/Noverse0/SIG-Net) dùng KDD Cup 2015 và NAVER;
repo không có loader XuetangX. [MST-GCN](https://github.com/wudongze9/MST-GCN)
có loader XuetangX đọc `train_log.csv`, `test_log.csv`, `train_truth.csv`,
`test_truth.csv`. Bộ vừa tải phù hợp tên file và schema này, nhưng chưa có
checksum hoặc manifest của tác giả MST-GCN để khẳng định đúng bản thí nghiệm.

Project lưu dữ liệu đúng `configs/xuetangx.yaml`: `data/raw/xuetangx`.
MST-GCN đang hard-code `data/xuetangx` trong `src_xuet/main.py`; cần truyền/chỉnh
đường dẫn trong adapter khi triển khai training. Lần này chỉ tải và thống kê,
chưa sửa code trong `third_party` hoặc chạy mô hình.

## Các khóa nối cần nhớ

| Bảng trái | Field | Bảng phải | Field |
|---|---|---|---|
| train/test_log | enroll_id | train/test_truth | enroll_id |
| train/test_log | username | user_info | user_id |
| train/test_log | course_id | course_info | course_id |

Trong file tải thực tế, `log.course_id` là **mã chuỗi** `course-v1:...`, nối với
`course_info.course_id`. Trang nguồn mô tả bộ dự đoán dùng mã số `course_info.id`,
nhưng mô tả này khác nội dung CSV thực tế. Giữ ID dạng chuỗi khi xử lý.
`truth=1` là bỏ học, `truth=0` là không bỏ học. `course_type=0` là học theo
lịch giảng viên, `course_type=1` là học theo tiến độ cá nhân.

Script thống kê không gộp train/test, không điền thiếu và không tự tạo nhãn.
Các ô rỗng được tính là thiếu; những chuỗi đặc biệt khác được giữ nguyên.
Tần suất và distinct được tính trên toàn bộ file, không ước lượng từ mẫu.

## Kiểm tra nhãn lần tải ngày 16/09/2026

| Tập | Lượt đăng ký | Bỏ học (1) | Không bỏ học (0) | Tỷ lệ bỏ học |
|---|---:|---:|---:|---:|
| Train | 157.943 | 119.817 | 38.126 | 75,8609% |
| Test | 67.699 | 51.316 | 16.383 | 75,8002% |

Mỗi file truth có `enroll_id` duy nhất và không có `enroll_id` trùng giữa
hai file truth. Kiểm tra này chưa khẳng định người học hoặc khóa học không
giao nhau giữa hai tập.
