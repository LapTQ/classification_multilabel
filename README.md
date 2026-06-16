# Multi-Label Classification

## 1. Cấu trúc thư mục

```text
classification_multilabel/
├── configs/
│   └── v1.multilabel.yaml       # File config
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
path/to/image/1.jpg,paperbag,shoulderbag
path/to/image/2.jpg,cart,handtrunk,plasticbag
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

### Đánh giá (Evaluate)

Chỉnh sửa `CKPT_PATH` trong file `src/entrypoints/eval.py` trỏ tới checkpoint, sau đó chạy:
```bash
python3 -m src.entrypoints.eval
```

### Dự đoán (Predict)

Chỉnh sửa cấu hình trực tiếp ở đầu file `src/entrypoints/predict.py`:
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

## 4. Lưu ý

### Thay đổi model
Chỉnh sửa trong file `src/core/model.py`: 
```python
# Mặc định sử dụng EfficientNet V2 S
self.model = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)
in_features = self.model.classifier[1].in_features
self.model.classifier[1] = nn.Linear(in_features, num_classes)

# Hoặc chuyển sang sử dụng ResNet50
# self.model = models.resnet50(
#     weights=models.ResNet50_Weights.DEFAULT
# )
# in_features = self.model.fc.in_features
# self.model.fc = nn.Linear(in_features, num_classes)
```

### Nhất quán tiền xử lý dữ liệu khi Huấn luyện và Dự đoán
Các phép tiền xử lý khi train và val được định nghĩa ở biến `self.train_transform` và `self.val_transform` trong file `src/core/data.py`. Khi predict cần đảm bảo biến `transform` trong file `src/entrypoints/predict.py` cũng áp dụng các phép tiền xử lý phù hợp.

