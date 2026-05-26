# Multi-Label Classification Training Pipeline

Dự án này cung cấp pipeline huấn luyện, đánh giá và dự đoán cho bài toán phân loại đa nhãn (multi-label classification), được phát triển trên nền tảng PyTorch Lightning. Dự án được kế thừa và cấu trúc lại từ pipeline single-label classification hiện có.

## 1. Cấu trúc thư mục

```text
/home/laptq/classification_multilabel/
├── configs/
│   └── v1.multilabel.yaml       # File cấu hình huấn luyện
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── augmentations.py     # Các phép tăng cường ảnh (Albumentations, torchvision)
│   │   ├── data.py              # Xử lý dataset đa nhãn (MultiLabelDataset) & DataModule
│   │   ├── model.py             # Kiến trúc model, loss function (BCEWithLogitsLoss) & metrics
│   │   └── utils.py             # Các hàm bổ trợ (tạo thư mục chạy, trực quan hóa lô dữ liệu)
│   └── entrypoints/
│       ├── train.py             # Script bắt đầu huấn luyện
│       ├── eval.py              # Script đánh giá checkpoint trên tập val/test
│       └── predict.py           # Script chạy suy luận (predict) trên ảnh/thư mục/file txt
```

## 2. Định dạng dữ liệu (Data Format)

Dữ liệu đầu vào là các file text `.txt`. Mỗi dòng tương ứng với một ảnh và danh sách các nhãn phân tách bằng dấu phẩy:
```text
<đường_dẫn_ảnh_tuyệt_đối>,<class_id_1>,<class_id_2>,...
```
*Ví dụ:*
```text
path/to/image/1.jpg,6
path/to/image/2.jpg,1,3,9,10
```

## 3. Hướng dẫn sử dụng

Trước khi chạy các script, hãy kích hoạt môi trường ảo:
```bash
source .venv/bin/activate
```

### Huấn luyện (Train)

Chỉnh sửa cấu hình huấn luyện trong `configs/v1.multilabel.yaml`, sau đó chạy:
```bash
python3 -m src.entrypoints.train
```

*Trong quá trình huấn luyện:*
*   Mô hình sử dụng mạng **EfficientNet-V2-S** làm backbone (load pretrained weights).
*   Hàm mất mát: **BCEWithLogitsLoss** (thích hợp cho bài toán đa nhãn).
*   Chỉ số theo dõi chính: **Validation F1 Macro** (`val_f1_macro`).

### Đánh giá (Evaluate)

Chỉnh sửa `CKPT_PATH` trong file [src/entrypoints/eval.py](file:///home/laptq/classification_multilabel/src/entrypoints/eval.py) trỏ tới checkpoint tốt nhất của bạn, sau đó chạy:
```bash
python3 -m src.entrypoints.eval
```

Sau khi chạy xong, một biểu đồ dạng cột thống kê F1-Score, Precision và Recall cho từng class trên tập test sẽ được lưu lại dưới tên `test_class_metrics.png` tại thư mục của checkpoint.

### Dự đoán (Predict)

Chỉnh sửa cấu hình trực tiếp ở đầu file [src/entrypoints/predict.py](file:///home/laptq/classification_multilabel/src/entrypoints/predict.py):
*   `CKPT_PATH`: Đường dẫn tới checkpoint `.ckpt`.
*   `INPUT_PATH`: Có thể là 1 file ảnh, 1 thư mục chứa các ảnh hoặc 1 file `.txt` chứa danh sách đường dẫn ảnh.
*   `OUTPUT_PATH`: Tên file đầu ra (mặc định là `predictions.txt`).
*   `THRESHOLD`: Ngưỡng xác suất để nhận diện nhãn hoạt động (mặc định là `0.5`).

Chạy lệnh suy luận:
```bash
python3 -m src.entrypoints.predict
```

Kết quả dự đoán sẽ được lưu ở định dạng:
```text
<đường_dẫn_ảnh>\t<nhãn_1>:<xác_suất_1>,<nhãn_2>:<xác_suất_2>,...
```
*Lưu ý:* Nếu không có nhãn nào vượt qua ngưỡng xác suất đặt ra, hệ thống sẽ tự động lấy nhãn có xác suất cao nhất làm kết quả fallback.
