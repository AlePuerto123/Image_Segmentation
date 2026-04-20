import torch


def calculate_metrics(preds, targets, num_classes):

    preds = torch.argmax(preds, dim=1)

    metrics = {}

    for cls in range(num_classes):

        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        TP = (pred_cls & target_cls).sum().item()
        FP = (pred_cls & ~target_cls).sum().item()
        FN = (~pred_cls & target_cls).sum().item()
        TN = (~pred_cls & ~target_cls).sum().item()

        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[cls] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics