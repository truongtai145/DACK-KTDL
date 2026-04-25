# 📊 MACHINE LEARNING PROJECT – DATA ANALYSIS & MODELING

## 📌 Giới thiệu

Dự án này thực hiện các bước phân tích dữ liệu, trực quan hóa, huấn luyện mô hình học máy và đánh giá hiệu năng. Ngoài ra, dự án còn áp dụng các thuật toán gom cụm để so sánh với nhãn thực tế.

---

## 📁 1. Đọc dữ liệu và phân tích cơ bản

Dữ liệu được đọc vào bằng thư viện `pandas`.

### Các thông tin được trích xuất:

* **Kích thước dữ liệu**:

  * Số dòng (samples)
  * Số cột (features)

* **Kiểu dữ liệu**:

  * Xác định kiểu của từng thuộc tính (int, float, object,...)

* **Phân bố nhãn (label)**:

  * Đếm số lượng mỗi lớp

* **Thống kê mô tả (cho dữ liệu số)**:

  * Giá trị lớn nhất (max)
  * Giá trị nhỏ nhất (min)
  * Giá trị trung bình (mean)

---

## 📉 2. Trực quan hóa dữ liệu (PCA)

### Mục tiêu:

* Giảm số chiều dữ liệu
* Hiển thị dữ liệu trên không gian 2D

### Phương pháp:

* Chọn các thuộc tính **liên tục (numeric)**
* Chuẩn hóa dữ liệu bằng `StandardScaler`
* Giảm chiều bằng **PCA (Principal Component Analysis)**
* Vẽ biểu đồ scatter với màu sắc khác nhau cho từng nhãn

### Kết quả:

* Dữ liệu được biểu diễn rõ ràng hơn trên mặt phẳng 2D
* Có thể quan sát sự phân tách giữa các lớp

---

## 🤖 3. Huấn luyện và đánh giá mô hình

### Các mô hình sử dụng:

* K-Nearest Neighbors (KNN)
* Random Forest
* Support Vector Machine (SVM)

### Phương pháp đánh giá:

* **K-Fold Cross Validation (k = 10)**
* Độ đo sử dụng: **F1-score (macro)**

### Tinh chỉnh tham số:

* Sử dụng `GridSearchCV` để tìm tham số tối ưu cho từng mô hình

### Bảng so sánh:

| Mô hình       | F1-score  |
| ------------- | --------- |
| KNN           | (kết quả) |
| Random Forest | (kết quả) |
| SVM           | (kết quả) |

### Nhận xét:

* Mô hình có F1-score cao nhất được chọn là tốt nhất
* Random Forest thường cho kết quả ổn định trên nhiều dataset

---

## 🔍 4. Gom cụm dữ liệu (Clustering)

### Mục tiêu:

* Phân nhóm dữ liệu mà không sử dụng nhãn

### Các thuật toán:

* **K-Means**
* **DBSCAN**

### Thực hiện:

* Loại bỏ cột nhãn
* Áp dụng hai thuật toán để phân cụm

### Đánh giá:

* So sánh kết quả gom cụm với nhãn thật bằng:

  * **Adjusted Rand Index (ARI)**

### Nhận xét:

* K-Means phù hợp khi số cụm đã biết trước
* DBSCAN có khả năng phát hiện nhiễu (noise)
* Hiệu quả phụ thuộc vào đặc điểm dữ liệu và tham số

---

## ⚙️ Công nghệ sử dụng

* Python
* pandas
* numpy
* matplotlib
* scikit-learn

---

## ▶️ Cách chạy chương trình

1. Cài đặt thư viện:

```bash
pip install pandas numpy matplotlib scikit-learn
```

2. Chạy file Python:

```bash
MAGIC_Analysis.ipynb
```

---

## 📌 Kết luận

* Dữ liệu đã được phân tích và trực quan hóa hiệu quả
* Các mô hình học máy được đánh giá khách quan bằng cross-validation
* Random Forest là mô hình tốt nhất (trong đa số trường hợp)
* Gom cụm giúp hiểu thêm cấu trúc dữ liệu mà không cần nhãn

---



## 📎 Tài liệu tham khảo

* Scikit-learn Documentation
* Machine Learning cơ bản
