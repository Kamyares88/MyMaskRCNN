# Mask R-CNN from scratch (PyTorch)

Minimal end-to-end training and inference setup for instance segmentation using Mask R-CNN with a ResNet50-FPN backbone.

## Layout
- `config.py` – configuration dataclass.
- `dataset.py` – dataset reading images + JSON annotations with polygons.
- `transforms.py` – image/target transforms.
- `model.py` – model factory and loader.
- `train.py` – training script.
- `infer.py` – run inference on images and save overlays/JSON.
- `eval.py` – quick IoU@0.5 precision/recall evaluation.
- `notebooks/maskrcnn_walkthrough.ipynb` – end-to-end walkthrough.

## Expected data format
```
data/train/
  images/img_0001.jpg
  annotations/img_0001.json
```
Annotation JSON example:
```json
{
  "boxes": [[x1, y1, x2, y2]],
  "labels": ["object"],
  "polygons": [
    [[x, y], [x, y], [x, y]]
  ],
  "iscrowd": [0]
}
```
Polygons are rasterized into masks; supply multiple polygons per instance for disjoint regions.

## Quickstart
1. Install deps: `pip install -r requirements.txt`
2. Train: `python train.py --train-data data/train --val-data data/val --output-dir outputs`
3. Inference: `python infer.py --images path/to/image_or_dir --checkpoint outputs/model_epoch_10.pth`
4. Evaluate: `python eval.py --data data/val --checkpoint outputs/model_epoch_10.pth`

