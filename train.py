# Train.
import argparse
from datetime import datetime
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex
from model import get_model
from loss_function import FocalLoss, sIoU_loss, BoundaryLoss
from data_loader import BuildingDataset

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def rebuild_optimizer(model, lr, weight_decay, use_adamw=False):
    params = get_trainable_params(model)
    if use_adamw:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

def print_trainable_summary(model):
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable:,} / Total params: {total:,} ({trainable / total * 100:.2f}%)")

def backup_geom_proj(model):
    backbone = getattr(model, "backbone", None)
    geom = getattr(backbone, "geom_proj", None)
    return {
        "weight": geom.weight.detach().cpu().clone(),
        "bias": geom.bias.detach().cpu().clone(),
        "weight_requires_grad": geom.weight.requires_grad,
        "bias_requires_grad": geom.bias.requires_grad
    }

def disable_geom_proj(model):
    bias = -4.5
    backbone = getattr(model, "backbone", None)
    geom = getattr(backbone, "geom_proj", None)
    geom.bias.data.fill_(bias)
    geom.weight.requires_grad = False
    geom.bias.requires_grad = False
    print(f"geom_proj disabled (bias = {bias}).")

def restore_geom_proj(model, backup):

    backbone = getattr(model, "backbone", None)
    geom = getattr(backbone, "geom_proj", None)
    geom.weight.data.copy_(backup["weight"].to(geom.weight.device, dtype=geom.weight.dtype))
    geom.bias.data.copy_(backup["bias"].to(geom.bias.device, dtype=geom.bias.dtype))
    geom.weight.requires_grad = bool(backup.get("weight_requires_grad", True))
    geom.bias.requires_grad = bool(backup.get("bias_requires_grad", True))
    print("geom_proj restored from backup.")

def train_and_evaluate(
        train_img_dir, train_mask_dir,
        val_img_dir=None, val_mask_dir=None,
        test_img_dir=None, test_mask_dir=None,
        epochs=100, batch_size=4, lr=1e-4,
        weight_decay=1e-5, patience=30, lr_factor=0.5, lr_patience=5,
        stage1_epochs=10, stage2_epochs=10, save_dir="./runs/train",
        use_adamw=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    os.makedirs(save_dir, exist_ok=True)

    dataset_train = BuildingDataset(img_dir=train_img_dir, mask_dir=train_mask_dir)
    dataset_val = BuildingDataset(img_dir=val_img_dir, mask_dir=val_mask_dir,
                                  boundary_dir=None) if val_img_dir else dataset_train


    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = dataset_train.num_classes
    model = get_model(num_classes).to(device)

    geom_backup = backup_geom_proj(model)

    if hasattr(model.backbone, "swin"):
        freeze_module(model.backbone.swin)
    if hasattr(model.backbone, "fpn"):
        freeze_module(model.backbone.fpn)
    if hasattr(model, "head"):
        freeze_module(model.head)
    if hasattr(model.backbone, "cdc"):
        unfreeze_module(model.backbone.cdc)
    restore_geom_proj(model, geom_backup)
    try:
        model.backbone.geom_proj.weight.requires_grad = True
        model.backbone.geom_proj.bias.requires_grad = True
    except Exception:
        pass

    print(">>> Stage1")
    print_trainable_summary(model)

    optimizer = rebuild_optimizer(model, lr, weight_decay, use_adamw=use_adamw)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=lr_factor,
                                                           patience=lr_patience)

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    autocast_ctx = lambda: torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available())

    metrics_dict = {
        "OA": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "(Multiclass)Precision": Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "(Multiclass)Recall": Recall(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "(Multiclass)F1": F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "mIoU": JaccardIndex(task="multiclass", num_classes=num_classes).to(device),
        "IoU": JaccardIndex(task="binary").to(device),
        "Precision": Precision(task="binary").to(device),
        "Recall": Recall(task="binary").to(device),
        "F1": F1Score(task="binary").to(device),
    }

    focal_loss = FocalLoss()
    boundary_loss = BoundaryLoss()

    best_iou = 0.0
    trigger_times = 0

    total_stage1 = stage1_epochs
    total_stage2 = stage2_epochs

    for epoch in range(epochs):
        if epoch < total_stage1:
            stage = 1
        else:
            stage = 2

        if epoch == total_stage1:
            print(f"\n>>> Transition: entering Stage2 at epoch {epoch}")
            if hasattr(model.backbone, "swin"):
                unfreeze_module(model.backbone.swin)
            if hasattr(model.backbone, "fpn"):
                unfreeze_module(model.backbone.fpn)
            if hasattr(model, "head"):
                unfreeze_module(model.head)
            if hasattr(model.backbone, "cdc"):
                freeze_module(model.backbone.cdc)
            disable_geom_proj(model)
            optimizer = rebuild_optimizer(model, lr, weight_decay, use_adamw=use_adamw)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=lr_factor,
                                                                   patience=lr_patience)
            print("Stage2 setup done: Swin&FPN&Head unfrozen, CDC frozen, geom disabled.")
            print_trainable_summary(model)

        print(f"\nEpoch [{epoch + 1}/{epochs}] (Stage {stage})")
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", colour="red")
        for imgs, seg_masks, boundary_masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            seg_masks = seg_masks.to(device, dtype=torch.long, non_blocking=True)
            boundary_masks = boundary_masks.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad()

            with autocast_ctx():
                if stage == 1:
                    outputs, aux_loss = model(imgs, gt=boundary_masks)
                else:
                    outputs, aux_loss = model(imgs, gt=seg_masks)

                if stage == 1:
                    loss = aux_loss
                else:
                    loss = focal_loss(outputs, seg_masks) + sIoU_loss(pred=outputs, target=seg_masks) + 0.5 * boundary_loss(outputs,seg_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Avg Train Loss: {avg_loss:.6f}")

        model.eval()
        with torch.no_grad():
            for metric in metrics_dict.values():
                metric.reset()

            for imgs, seg_masks, _ in tqdm(val_loader, desc="Evaluating (val)", colour="green"):
                imgs = imgs.to(device, non_blocking=True)
                seg_masks = seg_masks.to(device, dtype=torch.long, non_blocking=True)
                outputs, _ = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                for k, metric in metrics_dict.items():
                    metric.update(preds, seg_masks)

            val_results = {k: metric.compute().item() for k, metric in metrics_dict.items()}

        print("Validation Metrics:")
        for k, v in val_results.items():
            print(f"{k}: {v:.4f}")

        current_monitor = val_results["IoU"]
        scheduler.step(current_monitor)

        if current_monitor > best_iou:
            best_iou = current_monitor
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            trigger_times = 0
            print(f"New best model saved at epoch {epoch + 1} | val IoU={best_iou:.4f}")
        else:
            trigger_times += 1
            print(f"No improvement on val IoU. trigger_times={trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} based on validation set.")
                torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))
                break

        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / 1024 ** 2
            print(f"GPU Memory (MB): {mem:.1f}")


def parse_args():

    parser = argparse.ArgumentParser(description="Building Segmentation Training")
    parser.add_argument("--train_img", type=str, default=".\Dataset\Massachusetts_dataset\train\image", help="path_to_training_image")
    parser.add_argument("--train_label", type=str, default=".\Dataset\Massachusetts_dataset\train\label", help="path_to_training_label")
    parser.add_argument("--val_img", type=str,default=".\Dataset\Massachusetts_dataset\val\label", help="path_to_validation_image")
    parser.add_argument("--val_label", type=str, default=".\Dataset\Massachusetts_dataset\val\label",help="path_to_validation_label")
    parser.add_argument("--save_dir", type=str, default="./saved_weights", help="path_to_validation_label")

    parser.add_argument("--epochs", type=int, default=150, help="total_epochs")
    parser.add_argument("--stage_1_epochs", type=int, default=3, help="training_times_in_phase_1")
    parser.add_argument("--stage_2_epochs", type=int, default=135, help="training_times_in_phase_2")
    parser.add_argument("--bs", type=int,default=2 ,help="batch_size_in_training")
    parser.add_argument("--lr", type=float, default=0.00005,help="learning_rate")
    parser.add_argument("--lr_patience", type=int, default=30, help="decrease_lr_rate")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="decrease_extend_of_lr_rate")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="weight_of_adam_optimizer")
    parser.add_argument("--scheduler", type=int, default=150, help="early_stopping_epochs")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)

    train_and_evaluate(
        train_img_dir=args.train_img,
        train_mask_dir=args.train_label,
        val_img_dir=args.val_img,
        val_mask_dir=args.val_label,

        epochs=args.epochs,
        batch_size=args.bs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        stage1_epochs=args.stage_1_epochs,
        stage2_epochs=args.stage_2_epochs,
        save_dir=save_dir,
    )

