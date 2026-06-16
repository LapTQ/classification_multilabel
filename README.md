# Multi-Label Classification

## 1. File Structure

```text
classification_multilabel/
├── configs/
│   └── v1.multilabel.yaml       
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── augmentations.py     
│   │   ├── data.py              
│   │   ├── model.py             
│   │   └── utils.py             
│   └── entrypoints/
│       ├── train.py             
│       ├── eval.py             
│       └── predict.py           
```

## 2. Data Format

Input data consists of text files (`.txt`). Each line corresponds to an image and its associated labels separated by commas:
```text
<absolute_image_path>,<class_name_1>,<class_name_2>,...
```
*Example:*
```text
path/to/image/1.jpg,paperbag,shoulderbag
path/to/image/2.jpg,cart,handtrunk,plasticbag
```

## 3. Usage

Install dependencies
```bash
pip install -r requirements.txt
```

### Training

Edit the training configurations in `configs/v2.resnet50.rap2+cia+cia_gen.change_class_order.yaml`, then run:
```bash
python3 -m src.entrypoints.train
```

Notes:
* **Backbone selection**: Modify in `src/core/model.py`:
    ```python
    # EfficientNet V2 S
    self.model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.DEFAULT
    )
    in_features = self.model.classifier[1].in_features
    self.model.classifier[1] = nn.Linear(in_features, num_classes)

    # Or ResNet50
    self.model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )
    in_features = self.model.fc.in_features
    self.model.fc = nn.Linear(in_features, num_classes)
    ```
* **Preprocess transform customization**: Modify `self.train_transform` and `self.val_transform` in `src/core/data.py`.

### Evaluation

Edit `CKPT_PATH` in `src/entrypoints/eval.py` to point to your checkpoint, then run:
```bash
python3 -m src.entrypoints.eval
```

### Prediction

Configure settings directly at the beginning of `src/entrypoints/predict.py`:
*   `CKPT_PATH`: Path to the `.ckpt` checkpoint.
*   `INPUT_PATH`: Can be a single image file, a directory containing images, or a `.txt` file containing a list of image paths.
*   `OUTPUT_PATH`: Name of the output file (default is `predictions.txt`).
*   `THRESHOLD`: Probability threshold for active labels (default is `0.5`).

Run the inference command:
```bash
python3 -m src.entrypoints.predict
```

Inference results will be saved in the format:
```text
<image_path>\t<label_1>:<probability_1>,<label_2>:<probability_2>,...
```

**Important notes**:
* If no label exceeds the probability threshold, the model will fallback to selecting the label with the highest probability.
* Make sure that the `transform` variable in `src/entrypoints/predict.py` applies compatible preprocessing transforms in `src/core/data.py`
