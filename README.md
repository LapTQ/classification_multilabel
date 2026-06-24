# Multi-Label Classification

## 1. File Structure

```text
classification_multilabel/
├── configs/
├── src/
│   ├── __init__.py
│   ├── backbone/
│   │   ├── efficientnetv2s.py
│   │   └── resnet50.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── augmentations.py     
│   │   ├── data.py              
│   │   ├── model.py             
│   │   └── utils.py             
│   ├── entrypoints/
│   │   ├── bootstrap.py
│   │   ├── train.py             
│   │   ├── eval.py             
│   │   └── predict.py
│   └── tools/
│       ├── convert_onnx.py
│       └── visualize_logs.py
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