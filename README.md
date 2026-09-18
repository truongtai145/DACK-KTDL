# 📊 MACHINE LEARNING PROJECT – DATA ANALYSIS, MODELING & DATA MINING

## 📌 Giới thiệu

Dự án thực hiện quy trình phân tích và khai phá dữ liệu bằng các kỹ thuật Machine Learning. Mục tiêu là tìm hiểu đặc điểm của bộ dữ liệu, trực quan hóa dữ liệu, xây dựng các mô hình phân loại và áp dụng thuật toán gom cụm để khám phá cấu trúc dữ liệu.

Dự án sử dụng bộ dữ liệu **MAGIC Gamma Telescope**, trong đó các mẫu dữ liệu được phân loại thành hai nhóm: Gamma (`g`) và Hadron (`h`).

Các nội dung chính bao gồm:

* Phân tích tổng quan và thống kê dữ liệu.
* Tiền xử lý và chuẩn hóa dữ liệu.
* Trực quan hóa dữ liệu bằng PCA.
* Huấn luyện và tối ưu các mô hình phân loại.
* Đánh giá mô hình bằng Cross Validation và F1-score.
* Gom cụm dữ liệu bằng K-Means và DBSCAN.
* So sánh kết quả phân loại và khai phá dữ liệu.

---

## 🎯 Mục tiêu dự án

1. Khám phá cấu trúc, đặc điểm và phân bố của dữ liệu.
2. Giảm số chiều dữ liệu để trực quan hóa trên không gian hai chiều.
3. Xây dựng các mô hình học máy có khả năng phân loại dữ liệu.
4. So sánh hiệu năng giữa các thuật toán KNN, Random Forest và SVM.
5. Khám phá cấu trúc dữ liệu bằng các thuật toán gom cụm không giám sát.
6. Đánh giá mức độ tương đồng giữa các cụm được tạo ra và nhãn thực tế.

---

## 📁 1. Đọc dữ liệu và phân tích cơ bản

Dữ liệu được đọc và xử lý bằng thư viện `pandas`.

### 1.1. Thông tin tổng quan

Các thông tin được trích xuất bao gồm:

* **Số lượng mẫu (Samples):** Tổng số dòng dữ liệu.
* **Số lượng thuộc tính (Features):** Số cột được sử dụng để mô tả mỗi mẫu.
* **Kiểu dữ liệu:** Xác định kiểu dữ liệu của từng thuộc tính như `int`, `float`, `object`.
* **Dữ liệu thiếu:** Kiểm tra số lượng giá trị bị thiếu trong từng cột.
* **Dữ liệu trùng lặp:** Xác định các bản ghi trùng nhau nếu có.

### 1.2. Phân bố nhãn

Thống kê số lượng mẫu thuộc từng lớp:

* `g`: Gamma.
* `h`: Hadron.

Việc phân tích phân bố nhãn giúp xác định mức độ cân bằng giữa các lớp, từ đó lựa chọn phương pháp đánh giá mô hình phù hợp.

### 1.3. Thống kê mô tả

Sử dụng các thống kê cơ bản để tìm hiểu đặc điểm của dữ liệu số:

| Thống kê           | Ý nghĩa            |
| ------------------ | ------------------ |
| Mean               | Giá trị trung bình |
| Min                | Giá trị nhỏ nhất   |
| Max                | Giá trị lớn nhất   |
| Standard Deviation | Độ lệch chuẩn      |
| Median             | Giá trị trung vị   |

### Kết quả mong đợi

* Nắm được cấu trúc tổng thể của bộ dữ liệu.
* Phát hiện các vấn đề về dữ liệu thiếu hoặc trùng lặp.
* Hiểu sự khác biệt về phân bố giữa các thuộc tính.
* Làm cơ sở cho bước tiền xử lý và xây dựng mô hình.

---

## 🧹 2. Tiền xử lý dữ liệu (Data Preprocessing)

Tiền xử lý là bước quan trọng nhằm đảm bảo dữ liệu phù hợp với các thuật toán Machine Learning.

### 2.1. Tách thuộc tính và nhãn

Dữ liệu được chia thành hai phần:

* **X:** Các thuộc tính đầu vào dùng để huấn luyện mô hình.
* **y:** Nhãn thực tế cần dự đoán.

Đối với bài toán phân loại MAGIC Gamma Telescope, nhãn gồm hai lớp Gamma và Hadron.

### 2.2. Kiểm tra dữ liệu

Thực hiện kiểm tra:

* Giá trị thiếu (Missing Values).
* Giá trị trùng lặp (Duplicate Values).
* Kiểu dữ liệu không phù hợp.
* Giá trị bất thường cần xem xét.

Các bước xử lý cụ thể cần căn cứ vào kết quả kiểm tra thực tế.

### 2.3. Chuẩn hóa dữ liệu

Sử dụng `StandardScaler` để đưa các thuộc tính số về cùng thang đo.

Công thức chuẩn hóa:

$$
z = \frac{x-\mu}{\sigma}
$$

Trong đó:

* \(x\): Giá trị ban đầu.
* \(\mu\): Giá trị trung bình của thuộc tính.
* \(\sigma\): Độ lệch chuẩn.
* \(z\): Giá trị sau chuẩn hóa.

Chuẩn hóa đặc biệt có ý nghĩa đối với KNN và SVM vì khoảng cách hoặc độ lớn của các thuộc tính có thể ảnh hưởng đến quá trình học.

**Lưu ý:** Khi đánh giá bằng Cross Validation, cần thực hiện chuẩn hóa bên trong từng fold, chẳng hạn bằng `Pipeline`, để tránh rò rỉ dữ liệu (Data Leakage).

---

## 📉 3. Trực quan hóa dữ liệu bằng PCA

### 3.1. Mục tiêu

PCA (Principal Component Analysis) là phương pháp giảm chiều dữ liệu, giúp biểu diễn dữ liệu nhiều chiều trên không gian hai chiều.

Mục tiêu:

* Giảm số chiều của dữ liệu.
* Hạn chế sự phức tạp khi trực quan hóa.
* Quan sát sự phân bố và mức độ phân tách giữa các lớp.
* Hỗ trợ khám phá các đặc điểm nổi bật của dữ liệu.

### 3.2. Quy trình thực hiện

1. Lựa chọn các thuộc tính số.
2. Chuẩn hóa dữ liệu bằng `StandardScaler`.
3. Áp dụng PCA với hai thành phần chính.
4. Biểu diễn dữ liệu bằng biểu đồ Scatter Plot.
5. Sử dụng màu sắc khác nhau để phân biệt các nhãn.

### 3.3. Phương sai giải thích

Hai thành phần chính đầu tiên có tỷ lệ phương sai giải thích:

| Thành phần            | Explained Variance Ratio |
| --------------------- | -----------------------: |
| Principal Component 1 |                   42.24% |
| Principal Component 2 |                   15.75% |
| **Tổng**              |               **57.99%** |

Hai thành phần chính đầu tiên giữ lại khoảng 57.99% phương sai của dữ liệu, giúp biểu diễn một phần đáng kể thông tin trên không gian hai chiều.

### 3.4. Nhận xét

* PCA giúp giảm số chiều dữ liệu và hỗ trợ quan sát trực quan.
* Có thể sử dụng biểu đồ để xem xét mức độ chồng lấn giữa hai lớp.
* Các điểm dữ liệu gần nhau trên biểu đồ có thể có đặc điểm tương đồng trong không gian PCA.
* PCA chỉ giữ lại một phần phương sai nên biểu đồ hai chiều không thể phản ánh toàn bộ cấu trúc dữ liệu ban đầu.

**Lưu ý:** Khả năng phân tách trên biểu đồ PCA không đồng nghĩa trực tiếp với hiệu năng phân loại của mô hình.

---

## 🤖 4. Huấn luyện và đánh giá mô hình Machine Learning

### 4.1. Các mô hình sử dụng

#### K-Nearest Neighbors (KNN)

KNN phân loại một mẫu mới dựa trên nhãn của các mẫu lân cận gần nhất.

Đặc điểm:

* Dễ hiểu và dễ triển khai.
* Dựa trên khoảng cách giữa các mẫu.
* Nhạy cảm với thang đo của thuộc tính.
* Hiệu năng phụ thuộc vào số lượng láng giềng và cách tính khoảng cách.

#### Random Forest

Random Forest là thuật toán học máy tổ hợp, kết hợp nhiều cây quyết định để đưa ra kết quả dự đoán.

Đặc điểm:

* Có khả năng mô hình hóa các mối quan hệ phi tuyến.
* Có thể xử lý nhiều thuộc tính đầu vào.
* Hạn chế sự phụ thuộc vào một cây quyết định đơn lẻ.
* Có thể cung cấp thông tin về mức độ quan trọng của các thuộc tính.

#### Support Vector Machine (SVM)

SVM tìm kiếm siêu phẳng phân tách các lớp dữ liệu với biên phân tách phù hợp.

Đặc điểm:

* Có thể xử lý bài toán phân loại nhị phân.
* Hỗ trợ các hàm kernel để mô hình hóa quan hệ phi tuyến.
* Nhạy cảm với việc lựa chọn tham số.
* Thường cần chuẩn hóa dữ liệu để đạt hiệu quả phù hợp.

### 4.2. Phương pháp đánh giá

Sử dụng **10-Fold Cross Validation** để đánh giá hiệu năng của các mô hình.

Quy trình:

1. Chia dữ liệu thành 10 phần.
2. Sử dụng 9 phần để huấn luyện.
3. Sử dụng phần còn lại để đánh giá.
4. Lặp lại quá trình cho đến khi mỗi phần được dùng làm tập đánh giá một lần.
5. Tổng hợp kết quả từ các lần đánh giá.

Phương pháp này giúp đánh giá mô hình trên nhiều cách chia dữ liệu khác nhau.

Đối với bài toán phân loại có hai lớp, có thể sử dụng `StratifiedKFold` để duy trì tỷ lệ nhãn giữa các fold.

### 4.3. Độ đo đánh giá F1-score (Macro)

F1-score là độ đo kết hợp giữa Precision và Recall.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

Macro F1 được tính bằng trung bình cộng F1-score của từng lớp:

$$
F1_{\text{macro}} =
\frac{1}{C}\sum_{i=1}^{C} F1_i
$$

Trong đó \(C\) là số lượng lớp.

Macro F1 giúp đánh giá hiệu năng trên từng lớp với trọng số bằng nhau, kể cả khi số lượng mẫu giữa các lớp khác nhau.

### 4.4. Tinh chỉnh tham số

Sử dụng `GridSearchCV` để tìm kiếm tổ hợp tham số phù hợp cho từng mô hình.

Một số tham số có thể xem xét:

| Mô hình       | Tham số                                          |
| ------------- | ------------------------------------------------ |
| KNN           | `n_neighbors`, `weights`, `metric`               |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| SVM           | `C`, `kernel`, `gamma`                           |

Việc lựa chọn tham số được thực hiện dựa trên điểm đánh giá của tập validation trong quá trình tìm kiếm.

Để tránh rò rỉ dữ liệu, bước chuẩn hóa và tìm kiếm tham số nên được kết hợp trong quy trình Cross Validation.

---

## 📊 5. Kết quả và so sánh mô hình

### 5.1. Bảng kết quả

| Mô hình       | F1-score (Macro) |
| ------------- | ---------------: |
| KNN           |            0.878 |
| Random Forest |            0.887 |
| SVM           |            0.845 |

### 5.2. Trực quan hóa kết quả

Có thể sử dụng biểu đồ cột để so sánh F1-score của ba mô hình.

Biểu đồ giúp quan sát sự khác biệt về điểm đánh giá giữa các thuật toán.

### 5.3. Nhận xét kết quả

* KNN đạt F1-score Macro là 0.878.
* Random Forest đạt F1-score Macro là 0.887.
* SVM đạt F1-score Macro là 0.845.
* Trong ba kết quả được báo cáo, Random Forest có F1-score Macro cao nhất.
* KNN có kết quả gần với Random Forest.
* SVM có F1-score thấp hơn hai mô hình còn lại trong kết quả hiện tại.

Random Forest được xác định là mô hình có điểm F1-score cao nhất trong bảng so sánh. Tuy nhiên, để kết luận về mức độ ổn định và khả năng tổng quát hóa, cần xem xét thêm độ lệch chuẩn giữa các fold và kết quả trên tập kiểm tra độc lập.

---

## 🔍 6. Gom cụm dữ liệu (Clustering)

### 6.1. Mục tiêu

Gom cụm là kỹ thuật học máy không giám sát, nhằm tìm kiếm cấu trúc và nhóm các mẫu dữ liệu có đặc điểm tương đồng mà không sử dụng nhãn thực tế trong quá trình phân cụm.

Trong dự án, hai thuật toán được sử dụng là K-Means và DBSCAN.

### 6.2. K-Means

K-Means phân chia dữ liệu thành một số cụm được xác định trước.

Nguyên lý:

1. Khởi tạo các tâm cụm.
2. Gán mỗi mẫu vào cụm có tâm gần nhất.
3. Tính lại tâm của từng cụm.
4. Lặp lại quá trình cho đến khi đạt điều kiện dừng.

Đặc điểm:

* Phù hợp khi có thể xác định trước số lượng cụm.
* Dễ triển khai và tương đối nhanh.
* Nhạy cảm với việc khởi tạo tâm cụm.
* Có thể bị ảnh hưởng bởi nhiễu và các cụm có hình dạng phức tạp.

### 6.3. DBSCAN

DBSCAN là thuật toán gom cụm dựa trên mật độ của các điểm dữ liệu.

Hai tham số quan trọng:

* `eps`: Bán kính lân cận để xác định các điểm gần nhau.
* `min_samples`: Số lượng điểm tối thiểu để hình thành vùng có mật độ đủ cao.

Đặc điểm:

* Không cần xác định trước số lượng cụm.
* Có khả năng phát hiện các cụm có hình dạng không đều.
* Có thể xác định các điểm nhiễu (Noise).
* Kết quả phụ thuộc vào tham số và mật độ phân bố dữ liệu.

### 6.4. Quy trình gom cụm

1. Loại bỏ cột nhãn thực tế khỏi dữ liệu đầu vào.
2. Lựa chọn các thuộc tính dùng để phân cụm.
3. Chuẩn hóa dữ liệu.
4. Áp dụng K-Means.
5. Áp dụng DBSCAN.
6. So sánh nhãn cụm dự đoán với nhãn thực tế bằng ARI.

**Lưu ý:** Nhãn thực tế chỉ được sử dụng ở bước đánh giá, không được đưa vào quá trình huấn luyện gom cụm.

### 6.5. Đánh giá bằng Adjusted Rand Index (ARI)

ARI đo mức độ tương đồng giữa hai cách phân chia dữ liệu thành các nhóm, đồng thời hiệu chỉnh theo mức độ tương đồng có thể xuất hiện ngẫu nhiên.

Giá trị ARI có thể nằm trong khoảng:

| Giá trị ARI | Ý nghĩa                                              |
| ----------- | ---------------------------------------------------- |
| Gần 1       | Hai cách phân chia có mức độ tương đồng cao          |
| Gần 0       | Mức độ tương đồng tương đương với kỳ vọng ngẫu nhiên |
| Nhỏ hơn 0   | Mức độ tương đồng thấp hơn kỳ vọng ngẫu nhiên        |

ARI không yêu cầu nhãn cụm phải có cùng tên với nhãn thực tế.

### 6.6. So sánh K-Means và DBSCAN

| Tiêu chí        | K-Means                       | DBSCAN                                  |
| --------------- | ----------------------------- | --------------------------------------- |
| Số cụm          | Cần xác định trước            | Không cần xác định trước                |
| Cơ sở gom cụm   | Khoảng cách đến tâm           | Mật độ điểm                             |
| Phát hiện nhiễu | Hạn chế                       | Có                                      |
| Hình dạng cụm   | Phù hợp với cụm tương đối gọn | Có thể phát hiện cụm hình dạng phức tạp |
| Tham số chính   | `n_clusters`                  | `eps`, `min_samples`                    |

### 6.7. Nhận xét

K-Means và DBSCAN tiếp cận bài toán phân nhóm dữ liệu theo hai nguyên lý khác nhau.

K-Means tập trung phân chia dữ liệu dựa trên khoảng cách đến tâm cụm, trong khi DBSCAN tìm kiếm các vùng có mật độ điểm đủ cao và có khả năng tách các điểm nhiễu.

Kết quả ARI giúp đánh giá mức độ tương đồng giữa cấu trúc cụm tìm được và nhãn Gamma/Hadron. Tuy nhiên, ARI không thay thế hoàn toàn các chỉ số đánh giá chất lượng gom cụm nội tại như Silhouette Score.

---

## 📈 7. Phân tích tổng hợp kết quả

Dự án tiếp cận dữ liệu từ hai góc độ chính:

### Học máy có giám sát (Supervised Learning)

Các mô hình KNN, Random Forest và SVM sử dụng nhãn thực tế để học cách phân loại dữ liệu.

Hiệu năng được đánh giá bằng F1-score Macro và phương pháp 10-Fold Cross Validation.

### Học máy không giám sát (Unsupervised Learning)

K-Means và DBSCAN được sử dụng để khám phá cấu trúc dữ liệu mà không đưa nhãn thực tế vào quá trình phân cụm.

Kết quả gom cụm được đối chiếu với nhãn thực tế bằng ARI.

### Ý nghĩa

Việc kết hợp hai phương pháp giúp phân tích dữ liệu theo nhiều khía cạnh:

* Đánh giá khả năng dự đoán nhãn của các mô hình phân loại.
* Khám phá cấu trúc nhóm tiềm ẩn trong dữ liệu.
* So sánh sự khác biệt giữa phân loại có giám sát và gom cụm không giám sát.
* Tìm hiểu mức độ phù hợp của các thuật toán đối với bộ dữ liệu MAGIC Gamma Telescope.

---

## ⚙️ 8. Công nghệ sử dụng

| Công nghệ        | Vai trò                                         |
| ---------------- | ----------------------------------------------- |
| Python           | Ngôn ngữ lập trình                              |
| pandas           | Đọc, xử lý và phân tích dữ liệu                 |
| NumPy            | Tính toán số học                                |
| Matplotlib       | Trực quan hóa dữ liệu                           |
| scikit-learn     | Tiền xử lý, PCA, huấn luyện và đánh giá mô hình |
| Jupyter Notebook | Xây dựng và thực thi quy trình phân tích        |

---

## 📂 9. Cấu trúc dự án

```text
MAGIC-ML-Project/
│
├── magic.csv
├── MAGIC_Analysis.ipynb
├── README.md
│
└── images/
    ├── pca_visualization.png
    └── model_comparison.png
```

*Lưu ý: Đây là cấu trúc thư mục tham khảo. Tên file và thư mục cần điều chỉnh theo dự án thực tế.*

---

## ▶️ 10. Hướng dẫn cài đặt và chạy chương trình

### Bước 1: Cài đặt Python

Cài đặt Python và Jupyter Notebook hoặc sử dụng môi trường Anaconda.

### Bước 2: Cài đặt thư viện

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Bước 3: Chuẩn bị dữ liệu

Đặt file `magic.csv` vào đúng thư mục mà Notebook sử dụng để đọc dữ liệu.

### Bước 4: Khởi chạy Jupyter Notebook

```bash
jupyter notebook
```

Mở file:

```text
MAGIC_Analysis.ipynb
```

### Bước 5: Thực thi chương trình

Chạy lần lượt các cell từ trên xuống dưới để thực hiện:

1. Đọc và phân tích dữ liệu.
2. Tiền xử lý và chuẩn hóa.
3. Trực quan hóa bằng PCA.
4. Huấn luyện và đánh giá mô hình.
5. Tinh chỉnh tham số.
6. Gom cụm và tính ARI.
7. Tổng hợp kết quả.

---

## ⚠️ 11. Hạn chế của dự án

Một số hạn chế cần xem xét:

* PCA hai chiều chỉ giữ lại một phần phương sai của dữ liệu.
* KNN nhạy cảm với cách chuẩn hóa và lựa chọn số lượng láng giềng.
* Random Forest có thể cần tinh chỉnh tham số để hạn chế hiện tượng quá khớp.
* SVM phụ thuộc vào lựa chọn kernel và các tham số.
* K-Means nhạy cảm với khởi tạo tâm cụm và số cụm được chọn.
* DBSCAN có thể gặp khó khăn khi dữ liệu có mật độ phân bố không đồng đều.
* Kết quả gom cụm có thể khác với nhãn thực tế do mục tiêu tối ưu của thuật toán không giống bài toán phân loại.
* F1-score và ARI phản ánh những khía cạnh khác nhau, vì vậy không nên so sánh trực tiếp hai độ đo này.

---

## 🚀 12. Hướng phát triển

Trong tương lai, dự án có thể được mở rộng theo các hướng sau:

1. **Phân tích dữ liệu chuyên sâu:** Xây dựng thêm biểu đồ phân bố, ma trận tương quan và phân tích ngoại lệ.
2. **Tối ưu mô hình:** Mở rộng không gian tìm kiếm tham số và so sánh nhiều cấu hình khác nhau.
3. **Đánh giá toàn diện:** Bổ sung Precision, Recall, Confusion Matrix và ROC-AUC.
4. **Đánh giá gom cụm:** Bổ sung Silhouette Score và phân tích ảnh hưởng của tham số DBSCAN.
5. **Trực quan hóa nâng cao:** Sử dụng biểu đồ PCA để hiển thị kết quả phân loại và gom cụm.
6. **Triển khai ứng dụng:** Xây dựng giao diện cho phép tải dữ liệu, chạy mô hình và hiển thị kết quả phân tích.

---

## 📝 13. Kết luận

Dự án đã xây dựng quy trình phân tích dữ liệu, trực quan hóa, huấn luyện mô hình học máy và khai phá dữ liệu trên bộ dữ liệu MAGIC Gamma Telescope.

Thông qua PCA, dữ liệu nhiều chiều được biểu diễn trên không gian hai chiều nhằm hỗ trợ quan sát cấu trúc dữ liệu. Ba mô hình KNN, Random Forest và SVM được đánh giá bằng 10-Fold Cross Validation với F1-score Macro.

Theo kết quả hiện có, Random Forest đạt F1-score Macro 0.887, KNN đạt 0.878 và SVM đạt 0.845.

Bên cạnh đó, K-Means và DBSCAN được sử dụng để khám phá cấu trúc cụm mà không sử dụng nhãn trong quá trình học. ARI được áp dụng để đối chiếu kết quả gom cụm với nhãn thực tế.

Qua dự án, các kỹ thuật phân tích dữ liệu, giảm chiều, phân loại và gom cụm được kết hợp trong một quy trình thực hành Machine Learning, tạo nền tảng cho những nghiên cứu và ứng dụng khai phá dữ liệu tiếp theo.
