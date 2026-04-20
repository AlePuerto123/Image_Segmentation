import torch
from torch.utils.data import DataLoader
import config
from dataset import SegmentationDataset
from model import get_model
from metrics import calculate_metrics

def evaluate():

    dataset = SegmentationDataset(config.TEST_IMAGES,config.TEST_MASKS)

    print(f"Test images found: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1)

    model = get_model()
    #Load weights
    model.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True))
    model.eval()

    #Dictionaries to accumulate TP/FP/FN/TN counts for each class
    total_tp = {c: 0 for c in range(config.NUM_CLASSES)}
    total_fp = {c: 0 for c in range(config.NUM_CLASSES)}
    total_fn = {c: 0 for c in range(config.NUM_CLASSES)}
    total_tn = {c: 0 for c in range(config.NUM_CLASSES)}

    correct = 0
    total   = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            preds_raw = model(imgs)
            preds     = torch.argmax(preds_raw, dim=1)

            correct += (preds == masks).sum().item()
            total   += masks.numel()

            for cls in range(config.NUM_CLASSES):
                pred_cls   = (preds == cls)
                target_cls = (masks == cls)
                total_tp[cls] += (pred_cls &  target_cls).sum().item()
                total_fp[cls] += (pred_cls & ~target_cls).sum().item()
                total_fn[cls] += (~pred_cls & target_cls).sum().item()
                total_tn[cls] += (~pred_cls & ~target_cls).sum().item()

    class_names = {0: "Background", 1: "Person", 2: "Car", 3: "Dog"}

    print(f"\nOverall Pixel Accuracy: {correct / total:.4f}")
    print("-" * 50)

    for cls in range(config.NUM_CLASSES):
        tp = total_tp[cls]; fp = total_fp[cls]
        fn = total_fn[cls]; tn = total_tn[cls]

        accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"Class {cls} ({class_names[cls]}):")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1       : {f1:.4f}")