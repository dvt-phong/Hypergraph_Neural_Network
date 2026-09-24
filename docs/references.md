# Nguồn tham khảo cho code và thiết kế mô hình

Project này không phải bản tái hiện nguyên trạng của SIG-Net, MST-GCN, CA-TFHN,
HGNN hoặc HSL. Code hiện tại kết hợp một số ý tưởng từ các công trình đó rồi
điều chỉnh cho pipeline XuetangX gồm enrollment features, hypergraph và dự đoán
dropout.

Các ghi chú dưới đây phân biệt ba mức liên hệ:

- **Chuyển thể công thức:** code dùng cùng dạng toán học nhưng được viết lại.
- **Tham khảo ý tưởng:** kiến trúc hoặc cách biểu diễn dữ liệu gợi ý thiết kế.
- **Code riêng:** cách cài đặt cụ thể của project, không có bản tương ứng trực
  tiếp trong repository được dẫn.

---

## 1. SIG-Net

**Bài báo**

Daeyoung Roh, Donghee Han, Daehee Kim, Keejun Han, and Mun Yong Yi. “SIG-Net:
GNN Based Dropout Prediction in MOOCs Using Student Interaction Graph.” In the
39th ACM/SIGAPP Symposium on Applied Computing, pages 29–37, 2024.

- DOI: https://doi.org/10.1145/3605098.3636002
- Code: https://github.com/Noverse0/SIG-Net
- GitHub license metadata: MIT

**Liên hệ với project**

SIG-Net biểu diễn tương tác giữa học viên, khóa học và course object bằng graph,
sau đó khai thác subgraph của một learner-course target. Project hiện tại tham
khảo cách nhìn này khi tạo Course và Object hyperedges, cũng như khi dựng local
graph cho validation/test target.

Project không dùng RGCN + Bi-LSTM giống SIG-Net và không sao chép cấu trúc
subgraph của repository này.

---

## 2. MST-GCN

**Bài báo**

Yongkang Duan and Xuewen Chen. “An Adaptive Multi-Scale Spatio-Temporal Graph
Network for Robust MOOC Dropout Prediction.” Scientific Reports, 2026.

- DOI: https://doi.org/10.1038/s41598-026-40502-w
- Code: https://github.com/wudongze9/MST-GCN
- GitHub license metadata: chưa khai báo tại thời điểm rà soát

**Liên hệ với project**

MST-GCN dùng heterogeneous interaction graph và mô hình hóa tín hiệu hành vi ở
nhiều thang thời gian. Project tham khảo cách tách thông tin hành vi và quan hệ
giữa learner, course, object khi thiết kế preprocessing, features và graph.

Project hiện tại không triển khai MST-RGCN, adaptive history fusion gate, GRU
hoặc chuỗi graph snapshots của MST-GCN.

---

## 3. CA-TFHN

Tên chuẩn là **CA-TFHN**: Classmates Augmented Time-Flow Hybrid Network.

**Bài báo**

Guanbao Liang, Zhaojie Qian, Shuang Wang, and Pengyi Hao. “MOOCs Dropout
Prediction via Classmates Augmented Time-Flow Hybrid Network.” In Neural
Information Processing, pages 405–416, 2023.

- DOI: https://doi.org/10.1007/978-981-99-8184-7_31
- Code: https://github.com/codeds27/CA-TFHN
- GitHub license metadata: chưa khai báo tại thời điểm rà soát

**Liên hệ với project**

CA-TFHN kết hợp learning activity với quan hệ classmates. Project tham khảo ý
tưởng này khi tạo Behavioral hyperedges từ các enrollment có vector hành vi gần
nhau.

Project không triển khai Time-Flow Hybrid Network, LSTM, self-attention hoặc
link-prediction module của CA-TFHN.

---

## 4. HGNN

**Bài báo**

Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. “Hypergraph
Neural Networks.” Proceedings of the AAAI Conference on Artificial
Intelligence, 33(01), pages 3558–3565, 2019.

- DOI: https://doi.org/10.1609/aaai.v33i01.33013558
- Code: https://github.com/iMoonLab/HGNN
- GitHub license metadata: MIT

**Liên hệ với project**

`5_model.py` chuyển thể phép lan truyền HGNN chuẩn hóa từ incidence matrix. Với
incidence matrix `H`, node degree `D_v` và hyperedge degree `D_e`, dạng không có
edge weight riêng được dùng trong project là:

```text
D_v^(-1/2) H D_e^(-1) H^T D_v^(-1/2) X Theta
```

Code được viết lại bằng sparse PyTorch operations để membership value trong
`H*` vẫn nhận gradient. Nó không phải bản sao file model trong iMoonLab/HGNN.

---

## 5. HSL

**Bài báo**

Derun Cai, Moxian Song, Chenxi Sun, Baofeng Zhang, Shenda Hong, and Hongyan Li.
“Hypergraph Structure Learning for Hypergraph Neural Networks.” Proceedings of
IJCAI, pages 1923–1929, 2022.

- DOI: https://doi.org/10.24963/ijcai.2022/267
- Code: https://github.com/pkualpha/HSL
- GitHub license metadata: chưa khai báo tại thời điểm rà soát

**Liên hệ với project**

Project tham khảo ba ý chính:

1. Encode node trên hypergraph ban đầu `H0`.
2. Học lại cấu trúc để tạo `H*`, sau đó encode lần hai.
3. Dùng contrastive consistency giữa representation trước và sau refinement.

`6_hsl.py` không tái hiện nguyên bản thuật toán trong bài báo. Các khác biệt
chính:

- Chọn hyperedge theo family và size bucket, không học hyperedge sampling bằng
  Gumbel-Softmax.
- Lấy positive/negative candidate nodes rồi tính membership score.
- Dùng sigmoid membership và top-r hoặc threshold.
- Giữ nguyên các hyperedge không được chọn trong epoch.
- Khôi phục membership để tránh isolated node.
- `7_losses.py` dùng symmetric node-level contrastive loss; đây không phải đúng
  nguyên dạng intra-hyperedge contrastive objective của HSL.

---

## 6. Đối chiếu theo source file

| File | Nguồn liên quan | Cách sử dụng |
|---|---|---|
| `2_preprocess.py` | SIG-Net, MST-GCN, CA-TFHN | Đối chiếu cách tổ chức interaction data; split và CSV là code riêng |
| `3_features.py` | SIG-Net, MST-GCN, CA-TFHN | Tham khảo temporal/behavior representation; 94 features là thiết kế riêng |
| `4_hypergraph.py` | HGNN, SIG-Net, MST-GCN, CA-TFHN | Điều chỉnh interaction/classmates ideas thành Course, Object, Behavioral hyperedges |
| `5_model.py` | HGNN, HSL | Chuyển thể HGNN propagation và luồng `H0 → Z0 → H* → Z*` |
| `6_hsl.py` | HSL | Phiên bản membership refinement đơn giản hóa |
| `7_losses.py` | HSL | Contrastive consistency được điều chỉnh thành symmetric node-level loss |
| `8_train.py` | SIG-Net, MST-GCN, CA-TFHN | Đối chiếu bài toán dropout; training protocol là code riêng |

`0_config.py` chỉ chứa cấu hình. `1_download.py` chỉ tải dataset từ các URL đã
khai báo nên không được gán nguồn thuật toán từ năm mô hình trên.

---

## 7. Ghi chú về license và attribution

Trạng thái được kiểm tra ngày 2026-09-24 qua metadata GitHub:

| Repository | License được GitHub nhận diện |
|---|---|
| `Noverse0/SIG-Net` | MIT |
| `iMoonLab/HGNN` | MIT |
| `wudongze9/MST-GCN` | Chưa khai báo |
| `pkualpha/HSL` | Chưa khai báo |
| `codeds27/CA-TFHN` | Chưa khai báo |

“Chưa khai báo” không có nghĩa là code thuộc public domain. Khi repository không
có license rõ ràng, project này chỉ dùng bài báo và repository để nghiên cứu,
đối chiếu ý tưởng; không mặc định có quyền sao chép source code.
