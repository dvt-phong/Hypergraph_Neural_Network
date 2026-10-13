# Kế hoạch so sánh baseline công bằng — bản 2 (XuetangX là chính)

Ngày lập: 2026-10-10. **Chưa code.** Thay cho `PLAN_BASELINE.md` bản 1.

Danh sách baseline theo em: **LR, SVM, CFIN, MST-GCN, SIG-Net, CA-TFHN**. Bỏ so với bản 1: HGB, MLP, DP-SCL. MLP chuyển sang bảng ablation (kịch bản H).

Mô hình của mình: **M** (Course + Object + User + self-loop, X 89 cột, HGNN + MLP), transductive (d5c7771), chọn epoch theo val AUC, 5 seed, ngưỡng 0.5.

---

## 1. Trả lời ngắn

> **Không ép X của em vào các baseline có pipeline riêng, nhưng cũng không để mỗi bài tự chạy theo giao thức riêng.**
> Mọi mô hình dùng **cùng dữ liệu thô, cùng split, cùng giao thức huấn luyện và đánh giá**. Mỗi baseline tự biến dữ liệu thô thành đầu vào theo cách của chính nó (feature, chuỗi thời gian, đồ thị).
> Riêng **LR và SVM** không có cách trích feature riêng, nên dùng **X 89 cột của em**.

Nói gọn để viết trong bài: *same information, same protocol, method-specific representation* (cùng thông tin, cùng giao thức, biểu diễn riêng của từng phương pháp).

## 2. Ba phương án và vì sao chọn phương án C

| | A. Ép mọi baseline dùng X của em | B. Mỗi bài giữ nguyên giao thức gốc | **C. Đề xuất** |
|---|---|---|---|
| Feature | X 89 cho tất cả | Của từng bài | Của từng bài (LR/SVM dùng X) |
| Split, chọn epoch, seed | Chung | Mỗi bài một kiểu | **Chung** |
| Vấn đề | Làm mất đóng góp của baseline; nhiều kiến trúc không nhận được X | Số liệu không so được với nhau, có bài rò rỉ nhãn | Cần sửa phần giao thức trong code gốc |

**Vì sao không chọn A:**
1. Đóng góp chính của các bài này nằm ở cách xây đầu vào: CFIN có context smoothing + attention giữa user và course; MST-GCN có đồ thị nhiều góc nhìn theo thời gian; CA-TFHN có LSTM trên chuỗi hành động × ngày + đồ thị bạn cùng lớp; SIG-Net có đồ thị sinh viên–object–khóa học. Thay đầu vào bằng X tức là chạy một mô hình khác. Reviewer sẽ nói baseline đã bị làm yếu đi.
2. Về kỹ thuật, X là **một vector gộp 89 số**. MST-GCN và CA-TFHN cần **chuỗi theo ngày cho từng loại hành động**, SIG-Net cần **đồ thị**. Muốn đưa X vào thì phải thiết kế lại mô hình.
3. Errica et al. (2020) cho mọi GNN dùng cùng node feature. Lý do là ở đó feature chỉ là đầu vào chung, không phải đóng góp của mô hình nào. Mình làm theo đúng tinh thần đó: chỗ nào đầu vào là chung (LR, SVM) thì dùng chung X; chỗ nào cách xây đầu vào là phương pháp thì giữ nguyên.

**Vì sao không chọn B** (đã kiểm tra trong code gốc ở `baseline/`, xem bộ nhớ audit 2026-10-04):

| Bài | Giao thức gốc | Hậu quả |
|---|---|---|
| MST-GCN | Gộp train + test chính thức, lấy 80/20 theo thứ tự dòng; fit scaler trên toàn bộ; 8 epoch cố định | 22,570 enrollment thuộc test chính thức rơi vào train |
| SIG-Net | Gán loại cạnh sinh viên theo nhãn dropout, lấy cả nhãn test (`graph_builder.py:274-295`) | Rò rỉ nhãn test |
| CA-TFHN | In chỉ số test mỗi epoch, không có val; scaler fit trên train + test | Chọn mô hình dựa trên test |
| CFIN | Train trên cả train + val, epoch 27 cố định, 1 seed | Dùng nhiều dữ liệu train hơn M; không có độ lệch chuẩn |

Cùng một bảng mà mỗi dòng theo một luật khác nhau thì phép so sánh không còn ý nghĩa (Shchur et al., 2018; Kapoor & Narayanan, 2023).

**Tách hai câu hỏi ra hai bảng:**
- **Bảng baseline** (phương án C): M có tốt hơn các phương pháp đã công bố không?
- **Bảng ablation** (cùng X): hypergraph đóng góp bao nhiêu? Kịch bản H (chỉ MLP trên X, không đồ thị) đã trả lời câu này. Cộng thêm LR/SVM trên cùng X thì thấy được phần đóng góp của riêng feature.

## 3. Bảy quy tắc chung (áp dụng cho mọi dòng trong bảng)

| # | Quy tắc | Cụ thể trên XuetangX | Căn cứ |
|---|---|---|---|
| R1 | **Cùng dữ liệu thô** | Cùng 225,642 enrollment. Log cắt về **ngày 0–34** như M (log thô có tới ngày 36). Cùng `user_info`, `course_info`. Không mô hình nào thấy nhiều thông tin hơn mô hình khác | Kapoor & Narayanan 2023 |
| R2 | **Cùng split theo `enroll_id`** | train 126,354 / val 31,589 / test 67,699, đọc từ file split của M | Errica 2020; Shchur 2018 |
| R3 | **Nhãn chỉ của train; thống kê chỉ trên train** | Mô hình đồ thị được dùng **feature** (không nhãn) của val/test, vì M transductive cũng vậy. Scaler, vocab, thống kê đếm chỉ fit trên train | Kapoor & Narayanan 2023 (rò rỉ do tiền xử lý trên cả train và test) |
| R4 | **Chọn epoch trên val, test một lần** | Early stopping theo val AUC, giới hạn số epoch; chỉ chấm test ở epoch tốt nhất | Errica 2020 (tách *model selection* và *model assessment*) |
| R5 | **Siêu tham số: mức tune ngang nhau** | Deep baseline giữ siêu tham số mặc định của bài gốc (CFIN, MST-GCN `src_xuet`, CA-TFHN đều đã có bản XuetangX); chỉ thay số epoch cố định bằng R4. LR/SVM: chọn C trên val trong lưới 5 giá trị. M không tune thêm | Dacrema 2019; Shchur 2018 (baseline tune kém tạo ra tiến bộ giả) |
| R6 | **5 seed, cùng chỉ số** | Mean ± std của AUC, AUPRC, F1, Acc, P, R; ngưỡng 0.5 (`config.THRESHOLD`) | Shchur 2018 (nhiều lần khởi tạo); Gardner & Brooks 2018 (không so một con số đơn lẻ) |
| R7 | **Sửa lỗi, không sửa phương pháp** | Chỉ sửa lỗi code, lỗi định dạng dữ liệu và rò rỉ. Giữ nguyên kiến trúc, loss và feature. Mỗi bài có một file `*_changes.patch` (như CFIN đã làm) và ghi vào phụ lục | Dacrema 2019 (công bố code và cấu hình baseline) |

## 4. Áp dụng cho từng baseline

| Baseline | Đầu vào | Phải sửa để theo R1–R7 | Ghi chú |
|---|---|---|---|
| **LR** | X 89 | — | sklearn `LogisticRegression`, chọn C trên val. Gần như tất định, std ≈ 0 |
| **SVM** | X 89 | — | SVM kernel RBF trên 126k dòng tốn bộ nhớ và thời gian bậc hai (O(n²)), không chạy được. Đề xuất `LinearSVC` (chọn C trên val). AUC/AUPRC tính trên decision score; nhãn dự đoán lấy score ≥ 0 |
| **CFIN** | `feat_extract` gốc | Train chỉ trên train của M (không gộp val); early stopping theo val; 5 seed; scaler fit trên train; sửa lỗi gender (`'m'/'f'` trong code, dữ liệu là `'male'/'female'`) để gender không còn bằng 0 với mọi người | Kết quả cũ 0.8569 không dùng được vì khác giao thức, phải chạy lại |
| **MST-GCN** | Chuỗi ngày × hành động + đồ thị gốc | Bỏ chia 80/20 theo thứ tự dòng, dùng split của M; thêm val + early stopping thay 8 epoch cố định; scaler fit trên train; cửa sổ 35 ngày | Cần DGL, chạy trên server |
| **SIG-Net** | Đồ thị sinh viên–object–khóa học gốc | Loại cạnh chỉ dựa trên nhãn **train**; cạnh của val/test gán loại "chưa biết"; thêm early stopping | Bản gốc chỉ có KDD Cup / NAVER, cần adapter XuetangX (đã có bản cũ `src/12_signet.py` ở tag `full-hsl`). Cần DGL |
| **CA-TFHN** | Chuỗi hành động × ngày + đồ thị bạn cùng lớp gốc | Chọn epoch trên val thay cho test; scaler và thống kê đếm fit trên train; sửa lỗi gender giống CFIN | Đã chạy được trên CPU |

## 5. Bảng kết quả dự kiến

**Bảng 1 — So sánh với các phương pháp đã công bố** (XuetangX, test 67,699, 5 seed)

| Loại | Mô hình | Đầu vào | AUC | AUPRC | F1 |
|---|---|---|---|---|---|
| Truyền thống | LR | X | | | |
| | SVM (linear) | X | | | |
| Dropout MOOC | CFIN | riêng | | | |
| | CA-TFHN | riêng | | | |
| | MST-GCN | riêng | | | |
| | SIG-Net | riêng | | | |
| **Của mình** | **M** | X + hypergraph | | | |

Số liệu trong bài gốc chỉ ghi ở chú thích (khác split, khác giao thức).

**Bảng 2 — Ablation** (đã có kịch bản): M, H (bỏ hypergraph), A1–A4, …

## 6. Đoạn mẫu để viết trong bài

> *For a fair comparison, all methods use the same raw XuetangX records (logs of days 0–34, user and course information), the same enrollment-level train/validation/test split, and the same protocol: model selection by validation AUC with early stopping, a single evaluation on the test set, and five random seeds. Published baselines keep their own feature extraction and architecture, since input construction is part of their contribution; LR and SVM, which have no feature pipeline of their own, use our feature set. We only fixed bugs and label leakage in the released code (Appendix X); hyperparameters follow the original papers.*

## 7. Rủi ro cần nói trước

1. **Baseline thường ra thấp hơn số trong bài gốc**, vì không còn chọn mô hình trên test, không còn rò rỉ nhãn và không train trên val. Đây là điều đã lường trước; giải thích bằng bảng ở mục 2.
2. **Không có gradient boosting.** Trên dữ liệu bảng, GBDT thường mạnh ngang hoặc hơn deep learning (Grinsztajn et al., NeurIPS 2022), nên reviewer hay hỏi. Thầy nhớ là bài CFIN cũng so với GBDT, em kiểm tra lại bảng trong bài để chắc. Chạy HGB trên X chỉ mất vài phút.
3. **SIG-Net chưa từng chạy trên XuetangX trong bài gốc**, nên siêu tham số mặc định có thể chưa hợp. Ghi rõ điều này trong bài.

## 8. Em cần quyết định

- **Q1. CFIN với log1p hay bản gốc z-score?** Thầy đề xuất **log1p**. Lý do: đã chạy, mạnh hơn (+0.0029), cùng cách scale với M. Nguyên tắc ở đây là thay đổi nào giúp baseline mạnh hơn thì giữ và ghi rõ. Nếu em muốn theo R7 thật chặt thì dùng z-score gốc.
- **Q2. Sửa lỗi gender trong CFIN/CA-TFHN?** Thầy đề xuất **sửa** (R7: đây là lỗi định dạng dữ liệu, không phải phương pháp).
- **Q3. SVM:** `LinearSVC` (đề xuất) hay RBF xấp xỉ bằng Nyström (chậm hơn, thêm một siêu tham số)?
- **Q4. Thêm lại HGB** làm dòng truyền thống thứ ba? (rủi ro 2)
- **Q5. Thứ tự làm:** LR, SVM → CFIN, CA-TFHN (chạy CPU) → MST-GCN, SIG-Net (server, DGL). OULAD làm sau, theo đúng 7 quy tắc này.

---

## Tài liệu tham khảo

- Errica, Podda, Bacciu, Micheli. *A Fair Comparison of Graph Neural Networks for Graph Classification.* ICLR 2020. https://openreview.net/forum?id=HygDF6NFPB
- Shchur, Mumme, Bojchevski, Günnemann. *Pitfalls of Graph Neural Network Evaluation.* arXiv:1811.05868, 2018. https://arxiv.org/abs/1811.05868
- Ferrari Dacrema, Cremonesi, Jannach. *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys 2019. https://arxiv.org/abs/1907.06902
- Kapoor, Narayanan. *Leakage and the Reproducibility Crisis in Machine-Learning-Based Science.* Patterns 4(9), 2023. https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9
- Gardner, Brooks. *Evaluating Predictive Models of Student Success: Closing the Methodological Gap.* Journal of Learning Analytics 5(2), 2018. https://doi.org/10.18608/jla.2018.52.7
- Feng, Tang, Liu. *Understanding Dropouts in MOOCs* (CFIN). AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301517
- Grinsztajn, Oyallon, Varoquaux. *Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?* NeurIPS 2022 Datasets & Benchmarks. (thầy chưa kiểm tra lại link)
