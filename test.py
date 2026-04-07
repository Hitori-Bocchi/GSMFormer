# Test.
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex
from model import get_model

class BoundaryIoUEvaluator:
    def __init__(self, ignore_index=255, pixel_radius=3):
        self.ignore_index = ignore_index
        self.pixel_radius = pixel_radius
        k_size = 2 * self.pixel_radius + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    def _get_boundary(self, mask_np):
        mask_uint8 = mask_np.astype(np.uint8)
        eroded = cv2.erode(mask_uint8, self.kernel, iterations=1)
        boundary = mask_uint8 - eroded
        return boundary

    def compute_single_image(self, pred_mask, gt_mask):

        pred_binary = (pred_mask == 1).astype(np.uint8)
        gt_binary = (gt_mask == 1).astype(np.uint8)

        pred_boundary = self._get_boundary(pred_binary)
        gt_boundary = self._get_boundary(gt_binary)

        intersection = np.sum(np.logical_and(pred_boundary == 1, gt_boundary == 1))
        union = np.sum(np.logical_or(pred_boundary == 1, gt_boundary == 1))

        if union == 0:
            return np.nan

        return (intersection / union) * 100.0

class BuildingDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted([x for x in os.listdir(img_dir) if not x.startswith('.')])
        self.masks = sorted([x for x in os.listdir(mask_dir) if not x.startswith('.')])
        assert len(self.images) == len(
            self.masks), f"Image ({len(self.images)}) and mask ({len(self.masks)}) count mismatch!"
        self.num_classes = 2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        mask = np.array(mask, dtype=np.uint8)
        mask[mask > 0] = 1
        mask = torch.from_numpy(mask).long()

        return image, mask

def evaluate(
        test_img_dir,
        test_mask_dir,
        model_path,
        batch_size=1,
        num_classes=2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_test = BuildingDataset(img_dir=test_img_dir, mask_dir=test_mask_dir)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Test Dataset loaded. Images: {len(dataset_test)}")

    model = get_model(num_classes).to(device)

    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]

    model.load_state_dict(state)

    print("Model weights loaded")

    model.eval()

    metrics_dict = {
        "OA": Accuracy(task="multiclass", num_classes=2).to(device),
        "(Multiclass)Precision": Precision(task="multiclass", num_classes=2, average="macro").to(device),
        "(Multiclass)Recall": Recall(task="multiclass", num_classes=2, average="macro").to(device),
        "(Multiclass)F1": F1Score(task="multiclass", num_classes=2, average="macro").to(device),
        "mIoU": JaccardIndex(task="multiclass", num_classes=2).to(device),
        "IoU": JaccardIndex(task="binary").to(device),
        "Precision": Precision(task="binary").to(device),
        "Recall": Recall(task="binary").to(device),
        "F1": F1Score(task="binary").to(device),
    }

    biou_evaluator = BoundaryIoUEvaluator(pixel_radius=3)
    boundary_iou_scores = []

    with torch.no_grad():
        for metric in metrics_dict.values():
            metric.reset()

        for imgs, masks in tqdm(test_loader, desc="Evaluating", colour="blue"):
            imgs = imgs.to(device)
            masks = masks.to(device, dtype=torch.long)

            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1)

            for k, metric in metrics_dict.items():
                metric.update(preds, masks)

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(preds_np.shape[0]):
                score = biou_evaluator.compute_single_image(preds_np[i], masks_np[i])
                if not np.isnan(score):
                    boundary_iou_scores.append(score)

        test_results = {k: metric.compute().item() for k, metric in metrics_dict.items()}

    mean_biou = np.nanmean(boundary_iou_scores) if len(boundary_iou_scores) > 0 else 0.0

    for k, v in test_results.items():
        val_str = f"{v:.4f}"
        print(f"{k:<25}: {val_str}")
    print(f"{'Boundary IoU (3px)':<25}: {mean_biou:.4f} %")

def parse_args():
    parser = argparse.ArgumentParser(description="Building Segmentation Testing")
    parser.add_argument("--test_img", type=str, default=r".\Massachusetts_dataset\test\image",
                        help="path_to_testing_image")
    parser.add_argument("--test_label", type=str, default=r".\Massachusetts_dataset\test\label",
                        help="path_to_testing_label")
    parser.add_argument("--model_path", type=str, default=r".\saved_weights\Massachusetts\Massachusetts_tiny.pth",
                        help="path_to_tested_model")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    TEST_IMG_DIR = args.test_img
    TEST_MASK_DIR = args.test_label
    MODEL_PATH = args.model_path

    evaluate(
        test_img_dir=TEST_IMG_DIR,
        test_mask_dir=TEST_MASK_DIR,
        model_path=MODEL_PATH,
        batch_size=1,
        num_classes=2

    )
