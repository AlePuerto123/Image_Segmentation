import fiftyone.zoo as foz
import os
import numpy as np
from PIL import Image

import fiftyone.utils.openimages as _foi
import fiftyone.core.labels as _fol

_original_create_segmentations = _foi._create_segmentations
def _fixed_create_segmentations(*args, **kwargs):
    try:
        return _original_create_segmentations(*args, **kwargs)
    except TypeError:
        return _fol.Detections(detections=[])
_foi._create_segmentations = _fixed_create_segmentations


CLASS_MAP = {"Person": 1, "Car": 2, "Dog": 3}

paths = [
    "data/train/images", "data/train/masks",
    "data/val/images",   "data/val/masks",
    "data/test/images",  "data/test/masks",  
]
for p in paths:
    os.makedirs(p, exist_ok=True)

print("Downloading OpenImages with real segmentation masks...")

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["segmentations"],      
    classes=["Person", "Car", "Dog"],
    max_samples=500,                     
)

samples = list(dataset)
print(f"Total samples: {len(samples)}")


split_train = int(0.6 * len(samples))
split_val   = int(0.8 * len(samples))

splits = {
    "train": samples[:split_train],
    "val":   samples[split_train:split_val],
    "test":  samples[split_val:],
}

def make_mask(sample, width, height):
    """Converts a FiftyOne sample's segmentations into a class ID mask."""
    mask = np.zeros((height, width), dtype=np.uint8) 

    if sample.ground_truth is None:
        return mask

    for detection in sample.ground_truth.detections:
        label = detection.label
        if label not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[label]

        if detection.mask is None:
            continue

        
        x1 = int(detection.bounding_box[0] * width)
        y1 = int(detection.bounding_box[1] * height)
        w  = int(detection.bounding_box[2] * width)
        h  = int(detection.bounding_box[3] * height)
        x2 = min(x1 + w, width)
        y2 = min(y1 + h, height)

        det_mask = Image.fromarray(detection.mask.astype(np.uint8) * 255)
        det_mask = det_mask.resize((x2 - x1, y2 - y1), Image.NEAREST)
        det_mask = np.array(det_mask) > 0

        mask[y1:y2, x1:x2][det_mask] = class_id

    return mask

for split_name, split_samples in splits.items():
    print(f"Creating {split_name} split ({len(split_samples)} samples)...")
    for i, sample in enumerate(split_samples):
        img = Image.open(sample.filepath).convert("RGB")
        w, h = img.size

        img_dst  = f"data/{split_name}/images/{i}.jpg"
        mask_dst = f"data/{split_name}/masks/{i}.png"

        img.save(img_dst)

        mask = make_mask(sample, w, h)
        Image.fromarray(mask).save(mask_dst)

print("Dataset ready!")