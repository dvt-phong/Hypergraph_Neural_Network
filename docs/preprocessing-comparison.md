# Đối chiếu tiền xử lý SIG-Net, MST-GCN và CA-TFHN

Ngày kiểm tra: 16/09/2026. Đối chiếu trực tiếp code ở commit bên dưới và schema XuetangX đã tải trong project. Đây là audit code, chưa chạy toàn bộ pipeline hoặc training; không đồng nhất code công khai với pipeline thực nghiệm trong bài báo.

**Phạm vi:** trọng tâm là XuetangX của project. SIG-Net gốc không có loader XuetangX, nên cột SIG-Net mô tả nhánh KDD Cup 2015 để tham chiếu; nhánh NAVER được tách riêng. MST-GCN cũng có nhánh KDD riêng, không được nhầm với nhánh XuetangX.

## 1. Nguồn code và đường chạy thực sự

| Mã | Nguồn / commit | File và vị trí bằng chứng |
|---|---|---|
| S1 | SIG-Net `87bc320f04cc949347515d5f4c671a152e481e7e` | [utils.py](https://github.com/Noverse0/SIG-Net/blob/87bc320f04cc949347515d5f4c671a152e481e7e/src/utils.py#L44): `kddcup_load`, `naver_load` |
| S2 | Cùng commit | [graph_builder.py](https://github.com/Noverse0/SIG-Net/blob/87bc320f04cc949347515d5f4c671a152e481e7e/src/graph_builder.py#L28): KDD dòng 28–477; NAVER từ 480 |
| S3 | Cùng commit | [trainer.py](https://github.com/Noverse0/SIG-Net/blob/87bc320f04cc949347515d5f4c671a152e481e7e/src/trainer.py#L42): chọn train/test từ file gốc |
| M1 | MST-GCN `7350a7e9f3e9f042cc393997ade5921a2863c990` | [src_xuet/main.py](https://github.com/wudongze9/MST-GCN/blob/7350a7e9f3e9f042cc393997ade5921a2863c990/MST-GCN/src_xuet/main.py#L59): tạo `XuetangXGraph` |
| M2 | Cùng commit | [xuetangx_graph_builder.py](https://github.com/wudongze9/MST-GCN/blob/7350a7e9f3e9f042cc393997ade5921a2863c990/MST-GCN/src_xuet/xuetangx_graph_builder.py#L74): `process`, `_load_raw_data`, `__getitem__` |
| M3 | Cùng commit | [src_xuet/trainer.py](https://github.com/wudongze9/MST-GCN/blob/7350a7e9f3e9f042cc393997ade5921a2863c990/MST-GCN/src_xuet/trainer.py#L187): chia tập và khởi tạo model |
| M4 | Cùng commit | [src_kdd/graph_builder.py](https://github.com/wudongze9/MST-GCN/blob/7350a7e9f3e9f042cc393997ade5921a2863c990/MST-GCN/src_kdd/graph_builder.py#L28), [src_kdd/trainer.py](https://github.com/wudongze9/MST-GCN/blob/7350a7e9f3e9f042cc393997ade5921a2863c990/MST-GCN/src_kdd/trainer.py#L248) |
| C0 | CA-TFHN `f8ef659450458085603652e822285c6f2be5afca` | [repo tác giả](https://github.com/codeds27/CA-TFHN/tree/f8ef659450458085603652e822285c6f2be5afca); đã clone vào `third_party/CA-TFHN` |
| C1 | Cùng commit | [0_process_user_activity_logs.py](https://github.com/codeds27/CA-TFHN/blob/f8ef659450458085603652e822285c6f2be5afca/src/dataprocess/0_process_user_activity_logs.py#L10) |
| C2 | Cùng commit | [1_process_user_contextual_features.py](https://github.com/codeds27/CA-TFHN/blob/f8ef659450458085603652e822285c6f2be5afca/src/dataprocess/1_process_user_contextual_features.py#L10) |
| C3 | Cùng commit | [2_table_2_numpy.py](https://github.com/codeds27/CA-TFHN/blob/f8ef659450458085603652e822285c6f2be5afca/src/dataprocess/2_table_2_numpy.py#L23), [3_mapping.py](https://github.com/codeds27/CA-TFHN/blob/f8ef659450458085603652e822285c6f2be5afca/src/dataprocess/3_mapping.py#L8) |
| C4 | Cùng commit | [linkprediction_models.py](https://github.com/codeds27/CA-TFHN/blob/f8ef659450458085603652e822285c6f2be5afca/src/linkprediction/linkprediction_models.py#L22) |
| C5 | Cùng commit | [0_create_strong_classmates_graph.py](https://github.com/codeds27/CA-TFHN/blob/f8ef659450458085603652e822285c6f2be5afca/src/graphgeneration/0_create_strong_classmates_graph.py#L11), [1_graph_addition_information.py](https://github.com/codeds27/CA-TFHN/blob/f8ef659450458085603652e822285c6f2be5afca/src/graphgeneration/1_graph_addition_information.py#L14) |

MST-GCN có hàm `xuetangx_load()` trong `utils.py` đọc `user_profile.csv` và `course_info.csv`, **nhưng đường chạy XuetangX đang kiểm tra không gọi hàm này**. `main.py` gọi `XuetangXGraph`, rồi lớp này gọi `_load_raw_data()`. Vì vậy không thể kết luận mô hình dùng metadata chỉ từ việc thấy hàm trong utils.

## 2. Bảng tổng hợp pipeline

| Bước | SIG-Net — KDD Cup 2015 | MST-GCN — XuetangX | CA-TFHN — XuetangX |
|---|---|---|---|
| File log | `log_train.csv`, `log_test.csv` | `train_log.csv`; thêm `test_log.csv` nếu có | Cả `prediction_log/train_log.csv` và `test_log.csv` |
| File nhãn | `truth_train.csv`, `truth_test.csv`, không header; gán tên `enrollment_id, truth` | `train_truth.csv`; thêm `test_truth.csv` nếu có; đọc header | Hai truth CSV, đặt `enroll_id` làm index |
| Metadata | `date.csv`, `object.csv`, hai enrollment CSV | Không đọc `user_info.csv`, `course_info.csv`; tạo bảng ID từ log | Đọc cả `user_info.csv` và `course_info.csv` |
| Cách đọc | `pd.read_csv` toàn file; không chunks/usecols/dtype rõ ràng | Tương tự | Tương tự; chọn 5 cột sau `read_csv`, không phải `usecols` nên vẫn đọc file đầy đủ trước |
| Gộp train/test | Gộp log, nhãn, enrollment để tạo graph/feature | Gộp log và nhãn; biến `truth_train` sau gộp chứa cả hai tập | Gộp log/nhãn để tạo feature; gộp feature cho contextual/graph |
| Bỏ field | Không xóa hàng loạt; dùng tập cột cho từng bước. `source` không vào feature | `session_id` được đọc nhưng không dùng tiếp; metadata không được đọc | Chọn `enroll_id, username, course_id, action, time`; loại `session_id, object` khỏi bảng xử lý ngay |
| Mốc thời gian | Ngày lịch từ `time`, trừ `date.csv.from` của khóa học | Timestamp nhỏ nhất theo khóa học trên log đã gộp; không dùng ngày mở khóa học gốc | Ngày lịch của `time` trừ ngày lịch `course_info.start` |
| Cửa sổ | Ba snapshot ngày 0–9, 10–19, 20–29 | Gán ngày âm về 0; lọc `day < 35`; mốc ngày theo khoảng 24 giờ từ timestamp đầu | Đếm từng action ở `time_diff == 0..34`; sự kiện ngoài khoảng không đóng góp, không xóa toàn bộ dòng trước |
| Loại hành động | problem→0; video→1; discussion→2; wiki/page_close/navigate/access→3; lọc nhóm 0..3 khi dựng tương tác | Crosstab trên tất cả action còn sau lọc thời gian; không gộp về 4 nhóm | Danh sách cố định 22 action; giữ từng action riêng, không gộp 5 nhóm thành 5 feature |
| Lọc đăng ký | Enrollment dựa vào truth gộp; không thấy ngưỡng số click tối thiểu ở KDD | Loại dòng truth không nối được username/course_id; tùy chọn lấy mẫu enrollment | Inner join feature với truth; không có bước loại enrollment vì activity bằng 0 |
| Xử lý thiếu | KDD không có `dropna()` tổng quát; groupby có thể bỏ khóa NA mặc định | `dropna` chỉ trên truth sau join; `fillna(0)` cho feature tổng hợp; không có chính sách tường minh cho object rỗng | Không xóa người học thiếu hồ sơ; chuyển thông tin thiếu/sai về mã 0 theo từng hàm |
| Loại trùng | Groupby quan hệ để gom cạnh; không deduplicate log thô tổng quát | Deduplicate metadata enrollment, cạnh enrollment–object và object–course | Deduplicate bảng enrollment/user/course/mapping; không deduplicate sự kiện log trước đếm |
| Chuẩn hóa | StandardScaler click theo từng khóa học; StandardScaler feature khóa học; fit trên dữ liệu gộp | Fit enrollment aggregate, object action và course aggregate trên dữ liệu gộp; chuỗi đếm 35 ngày không scale ở bước này | Scale age và hai enrollment-count trên train+test; scale tiếp user/course feature ở nhánh graph; raw activity counts không scale ở script chuyển tensor |
| Chia tập dự đoán | Giữ ID từ truth_train/truth_test gốc | Cắt 80% index đầu / 20% cuối; **không shuffle trước cắt**; sampler chỉ xáo trộn bên trong từng phần | Tách lại theo membership enrollment gốc; `tt_label` tạo train/test mask cho graph |
| Đầu vào thời gian | Ba subgraph ứng với ba cửa sổ riêng | `__getitem__` trả cùng `subgraph` ba lần | 35 ngày × 22 action; bản NPZ reshape `N × 5 × 7 × 22`; graph lưu `N × 35 × 22` |
| Bằng chứng | S1; S2 dòng 33–175, 320–362; S3 dòng 42–55 | M2 dòng 94–212, 237–283, 390–400; M3 dòng 209–212 | C1 dòng 10–61; C2 dòng 12–86; C3; C5 |

**Lưu ý về “xóa”:** không dùng một field trong feature không đồng nghĩa xóa cột đó khỏi file CSV gốc. Các pipeline đọc dữ liệu rồi tạo bảng/tensor/graph mới; audit này không chỉnh sửa dữ liệu raw của project.

## 3. Từng field của XuetangX được dùng thế nào?

SIG-Net gốc không xử lý trực tiếp schema XuetangX, nên bảng field dưới đây chỉ có hai implementation thực sự đọc bộ này.

| Bảng.field | MST-GCN XuetangX | CA-TFHN XuetangX | Ý nghĩa đối với tiền xử lý |
|---|---|---|---|
| log.enroll_id | Map thành enrollment-node ID; groupby click/action; nối truth | Groupby 770 activity features; nối truth; map record-node | Khóa bắt buộc, không đưa ID nguyên dạng vào vector số |
| log.username | Bổ sung vào truth; nhóm các enrollment cùng người học; đếm số user/khóa học | Nối hồ sơ; đếm số enrollment/người; tạo user-node ID | Giữ để nối bảng/dựng quan hệ |
| log.course_id | Map course-node; suy ra start từ log; aggregate course | Nối start đúng bằng course_id; tạo khóa graph; lookup category ở bước sau có lỗi khóa | Giữ mã chuỗi nguyên dạng |
| log.session_id | Đọc nhưng không dùng | Loại ngay khỏi bảng chọn 5 cột | Cả hai không có session feature trong pipeline kiểm tra |
| log.action | Đếm phân bố action/enrollment và action/object | Đếm 22 action theo từng ngày | Trường nội dung hành vi chính |
| log.object | Tạo object-node và cạnh; đếm unique_objects; crosstab action | Loại ngay khỏi bảng xử lý | Không thể bỏ object nếu muốn giữ graph ba loại node của MST-GCN |
| log.time | Parse datetime, tạo day; raw timestamp không vào tensor cuối | Cắt giờ thành ngày lịch, tạo time_diff | Hai mô hình có mốc thời gian khác nhau |
| truth.enroll_id | Nối log metadata; map nhãn đúng thứ tự enrollment | Index/join với feature và xác định membership train/test | Không phải feature dự đoán |
| truth.truth | Nhãn float32 cho enrollment | Nhãn dự đoán; giữ trong graph.labels | Không được xem là field hành vi |
| user_info.user_id | Không đọc file | Khóa nối log.username | Profile bao phủ rộng hơn tập dự đoán |
| user_info.gender | Không dùng | Hàm dự kiến m→1, f→2, còn lại→0; CSV thực tế male/female đều về 0 | Code hiện làm mất thông tin giới tính của bản tải |
| user_info.education | Không dùng | Bachelor's→1, High→2, Master's→3, Primary→4, Middle→5, Associate→6, Doctorate→7; non-string→0 | Chuỗi ngoài danh sách gây ValueError, không có fallback cho chuỗi lạ |
| user_info.birth | Không dùng | age=2023−int(birth); thiếu hoặc age ngoài [10,70]→0; sau đó scale age | Không xóa người có năm sinh sai; mốc 2023 hard-code |
| course_info.id | Không dùng | Đặt index cho lookup category | Không trùng khóa course_id dạng chuỗi trong log thực tế |
| course_info.course_id | Không đọc; lấy course_id từ log | Nối start ở bước activity đúng; phần category không dùng khóa này | Cần phân biệt hai bước join khác nhau |
| course_info.start | Không đọc; thay bằng min(time) của khóa học | Dùng ngày mở khóa học để tính ngày tương đối | Có thể làm tập sự kiện được giữ khác MST-GCN |
| course_info.end | Không dùng | Không dùng trong các bước feature được kiểm tra | Không lọc log theo ngày kết thúc |
| course_info.course_type | Không dùng | Không dùng | Không có nhánh xử lý riêng IPM/SPM ở preprocessing này |
| course_info.category | Không dùng | Định mã 18 ngành từ 1..18; missing/non-string→0; nhưng lookup sai khóa với bản tải | Có danh sách mã hóa không có nghĩa feature đã nhận thông tin đúng |

Nguồn chi tiết: M2 dòng 162–203, 243–281; C1 dòng 10–23; C2 dòng 14–84. Đối chiếu field thực tế: `outputs/reports/xuetangx/statistics.md`.

## 4. Feature mới và phép biến đổi

| Mô hình | Feature/tensor | Cách tạo thực tế | Điểm cần chú ý |
|---|---|---|---|
| SIG-Net KDD | 30 `clicksum_day*` | Groupby enrollment/event/day/object, đếm time; sau đó gán vào ô ngày | Dòng 114 dùng `=` thay vì `+=`: nhiều nhóm cùng ngày ghi đè nhau; không được mô tả là tổng click/ngày đã cộng đúng |
| SIG-Net KDD | labels0/1/2 | Cửa sổ tương ứng không hoạt động →1, có hoạt động→0 | labels3 mới là truth gốc được dùng trong loss |
| SIG-Net KDD | Course vector 7 chiều | Số module thuộc video/problem/discussion/chapter/html/about + tổng enrollment | `category` ở đây là loại module KDD, khác ngành học trong XuetangX |
| SIG-Net KDD | Node vector 20 chiều | 3 cờ loại node + 10 giá trị ngày + 7 giá trị course, padding 0 tùy node | `object_feature` one-hot 4 event có được tính nhưng không được nối vào object node vector cuối |
| MST-GCN XuetangX | Enrollment vector | 35 daily counts + total_actions + unique_objects + A action counts; scale phần aggregate | Kích thước dự kiến 37+A; nếu A=22 thì 59. A được suy ra sau lọc, không hard-code 22 |
| MST-GCN XuetangX | Course vector 4 chiều | total_actions, num_enrollments, num_users, density=total_actions/num_enrollments | Loại start khỏi vector, scale 4 giá trị; không có category/course_type |
| MST-GCN XuetangX | Object vector A chiều | Crosstab object/action rồi StandardScaler | NA object không được làm sạch rõ ràng trước mapping/crosstab |
| MST-GCN XuetangX | Quan hệ graph | Enrollment↔object; object↔course; subgraph lấy thêm enrollment cùng user | friends.csv là tùy chọn user1/user2; bản dữ liệu tải không có file này |
| CA-TFHN | 770 activity counts | 35 ngày ×22 action; cộng indicator theo enrollment | Không gom 22 action thành 5 feature; ngày ngoài cửa sổ không đóng góp |
| CA-TFHN | 6 contextual fields trong NPZ | age, gender, education, user_enroll_num, course_enroll_num, course_category | Ba cột số age và hai count được chuẩn hóa |
| CA-TFHN | 7 chiều org_context ở graph | 5 user fields gồm thêm cluster_label + trung bình 2 course fields của các khóa user đã đăng ký | Khác tensor context 6 chiều của bước NPZ; pipeline thiếu bước sinh/gắn cluster_label |
| CA-TFHN | 32 chiều enhanced_context | Ghép embedding user 16 chiều với trung bình embedding course 16 chiều | Sinh từ link prediction trên user–course graph |
| CA-TFHN | Cạnh classmates | Ma trận sở thích dự đoán threshold 0,6, ép đăng ký đã có thành 1; nối hai record cùng khóa học nếu cosine≥0,95, bỏ đường chéo | Không phải cạnh chung object, vì object đã bị bỏ |

Các nhóm action CA-TFHN, giữ đúng thứ tự code:

| Nhóm tổ chức trong code | Các action |
|---|---|
| Video (5) | seek_video, play_video, pause_video, stop_video, load_video |
| Problem (6) | problem_get, problem_check, problem_save, reset_problem, problem_check_correct, problem_check_incorrect |
| Forum (5) | create_thread, create_comment, delete_thread, delete_comment, close_forum |
| Click (5) | click_info, click_courseware, click_about, click_forum, click_progress |
| Close (1) | close_courseware |

## 5. Các khác biệt/lỗi cần ghi nhận trước tái lập

| Mức xác nhận | Phát hiện | Hệ quả | Bằng chứng |
|---|---|---|---|
| Xác nhận từ đường gọi | SIG-Net gốc chỉ dispatch KDD hoặc NAVER; không có XuetangX | Chưa thể nói SIG-Net gốc đã bỏ field nào của XuetangX; muốn benchmark phải có adapter được mô tả riêng | `SIG-Net/src/main.py` dòng 12–15 |
| Xác nhận từ đường gọi | MST src_xuet trainer khởi tạo trực tiếp Multi_MST_GCN | Chọn tên SIG_Net ở tham số `--model` không chứng minh đã chạy SIG-Net; help/comment liệt kê baseline nhưng trainer kiểm tra không dispatch theo tên | M3 dòng 199–202 |
| Xác nhận từ code | MST day dựa min timestamp của khóa học, khác start chính thức | 35 ngày này không tương đương cửa sổ CA-TFHN; phải đồng nhất protocol khi so sánh | M2 dòng 270–281; C1 dòng 19–23 |
| Xác nhận từ code | MST bỏ split gốc, cắt lại 80/20 không shuffle | Với bản đủ 225.642 enrollment và nếu không bị lọc/mất mẫu: 180.513 train, 45.129 test; 22.570 nhãn từ test gốc đi vào train mới | M2 dòng 251–260; M3 dòng 209–212; các số là tính theo điều kiện, chưa chạy loader |
| Xác nhận từ code | MST trả cùng subgraph ba lần; window_size=35 trong lớp, không nhận CLI window_size ở main | Không được mô tả nhánh này tạo ba snapshot khác nhau; tham số CLI không thay đổi cửa sổ của lớp XuetangX hiện tại | M2 dòng 74–82, 390–391; M1 dòng 72–77 |
| Xác nhận cấu trúc code | Cách padding act_matrix dùng số cột hiện có thay vì reindex đủ ngày 0..34 | Nếu các ngày vắng nằm ở giữa, có thể ghi đè cột ngày có dữ liệu và lệch thứ tự; chưa đo phát sinh trên bản tải | M2 dòng 162–168 |
| Cần kiểm tra runtime | MST đưa `logs.object.unique()` vào map mà không dropna/fill sentinel | Object rỗng của dataset không được xử lý nhất quán giữa mapping/groupby/crosstab; không kết luận đã loại hết hay đã impute | M2 dòng 114–141, 201–203 |
| Xác nhận bằng hàm gốc + thống kê CSV | CA gender_convert(male)=0, gender_convert(female)=0 | Tất cả giới tính hợp lệ male/female trong bản CSV này bị mã hóa giống missing | C2 dòng 31–40; `preprocessing_checks.json` |
| Xác nhận khóa metadata | CA lookup category bằng index=id nhưng truyền log.course_id | Không tìm đúng ngành học cho mã chuỗi; tập id và course_id trong course_info tải về có giao bằng 0 | C2 dòng 60, 76–77; `preprocessing_checks.json` |
| Xác nhận từ code | CA age dùng 2023, không dùng thời điểm log 2015–2017 | Tuổi không phải tuổi tại thời điểm học; giá trị ngoài 10–70 bị đặt 0, không loại row | C2 dòng 18–24 |
| Xác nhận thiếu bước trong repo | CA các bước graph đòi cluster_label; các bước 0–2 không tạo cột này; repo có file cluster nhưng không thấy code gắn nhãn vào feature CSV | Pipeline từ raw chưa hoàn chỉnh; sẽ thiếu cột nếu chỉ chạy tuần tự các script đã công bố | C3 mapping dòng 8–16; C4 dòng 22–27; tìm toàn repo `cluster_label` |
| Xác nhận từ script | CA feat_extract.sh gọi `1_process_user_contextual.py`, tên file thật có `_features`; đường `../../datastore` phụ thuộc cwd | Lệnh shell trong README không đủ bảo đảm pipeline chạy được nguyên trạng | C0 `feat_extract.sh`; C2 dòng 8–9 |
| Xác nhận từ script | CA dump_data.sh wget vào cwd nhưng tar đọc `./datastore/prediction_data.tar.gz`; hai script lưu graph đặt `'wb'` vào os.path.join thay vì mode của open | Lỗi đường dẫn/lưu file cần sửa trước chạy | C0 dump_data.sh; C5 graph0 dòng 83, graph1 dòng 68 |

### Train/test và nguy cơ thông tin ngoài train

| Mô hình | Điều code đang làm | Cách diễn giải đúng |
|---|---|---|
| Cả ba pipeline so sánh | Fit scaler trên dữ liệu gộp, không chỉ training rows | Không phải quy trình inductive “fit train, transform test”; cần công bố hoặc sửa đồng nhất |
| MST/CA graph | Dùng hành vi/quan hệ của cả train và test khi dựng graph | Có thể là thiết lập transductive; sự hiện diện của test feature không tự động có nghĩa đã dùng nhãn test trong loss |
| SIG-Net KDD | Loại cạnh giữa enrollment cùng user dựa labels1/2/3; labels3 lấy từ truth_all gồm test; khi lấy subgraph có bước sửa loại cạnh xuất phát từ target | Đây là điểm cần audit label leakage kỹ: masking target không tự chứng minh đã loại mọi thông tin nhãn test ở các node khác. Chưa kiểm chứng runtime hay kết luận về kết quả bài báo (S2 dòng 274–299, 449–475) |
| CA link prediction | RandomLinkSplit val=0,1/test=0,1 cho **cạnh user–course**, với disjoint_train_ratio=0,3 | Không phải tỷ lệ chia lại các mẫu dropout; dropout vẫn theo tt_label gốc (C4 dòng 68–82; C5 dòng 42–47) |

## 6. Hai nhánh phụ để tránh áp dụng nhầm

### SIG-Net NAVER

| Bước | Hành vi | Vị trí S2 |
|---|---|---|
| Đọc | Gộp các sheet ở 7 workbook CS/Web/App/General/Design/AI/DS; sheet được đọc rộng hơn tập thực sự dùng | S1 `naver_load()` |
| Khóa học/người học | Loại course_id=12134; giới hạn log/lecture về danh sách course còn lại; user_info chỉ role_priority=2 | 487–500 |
| Log | Lấy video/evaluation/forum/comment; evaluation chỉ QUIZ; chuẩn hóa về course_id, ymdt, user_id, type, type_id | 502–522 |
| Missing | `log_all.dropna()` trước merge enrollment; đây là xóa dòng thiếu trên các cột đã chọn | 524–525 |
| Thời gian | Mốc là ngày log đầu của từng enrollment; giữ ngày 0..39 | 526–531 |
| Feature/nhãn | 30 ngày đầu làm feature; không hoạt động ngày 30..39 → dropout; bỏ enrollment không có activity trong 40 ngày | 537–566 |
| Chia tập | train_test_split test_size=0,40, random_state=321; không thấy stratify | 665–673 |

### MST-GCN KDD

| Bước | Khác nhánh XuetangX | Vị trí M4 |
|---|---|---|
| Nguồn | Đọc bộ file KDD giống SIG-Net; metadata date/object/enrollment | graph_builder 36–38 |
| Subset | Lấy mẫu enrollment seed=0; tối thiểu 1 nếu số tính ra là 0; lọc tiếp course metadata | 41–59 |
| Action | Map 4 nhóm như SIG-Net, action lạ/NA gán nhóm 3 | 71–74 |
| Thời gian | 3 snapshot, mỗi window_size ngày; default của lớp=10; mask đầu chỉ `days < window_size`, chưa chặn ngày âm | 29–33, 79–83 |
| Daily count | Pivot theo ngày với aggfunc=sum, reindex đủ ngày và fillna(0) | 114–119; khác phép gán ghi đè SIG-Net |
| Missing cạnh | Loại cạnh object-course thiếu dst; enrollment-object thiếu src/dst | 224, 240 |
| Split | Shuffle các index 1.. trước cắt 80/20; index 0 là full graph | trainer 254–262; khác XuetangX không shuffle trước cắt |

## 7. Áp dụng cho project hiện tại

Giữ nguyên `data/raw/xuetangx` và lưu mỗi bản xử lý trong `data/processed`. Nếu mục tiêu là so sánh công bằng, cần quyết định chung: split gốc hay split lại, mốc 35 ngày từ đâu, quyền dùng test feature trong graph, và scaler fit trên tập nào. Sau đó tách các feature riêng của từng model: MST cần object, CA bỏ object; không được xóa object khỏi bản dữ liệu dùng chung.

Chưa chạy pipeline để đếm chính xác số dòng bị loại sau cửa sổ thời gian. Báo cáo này mô tả điều kiện lọc trong code; các số của dataset gốc nằm trong `outputs/reports/xuetangx/statistics.md`. Không sửa code tác giả hoặc raw CSV trong lần audit này.
