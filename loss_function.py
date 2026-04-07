# Loss functions.
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def sIoU_loss(pred, target, smooth=1e-6):

    target = target.unsqueeze(1)

    pred = torch.sigmoid(pred)
    target = F.one_hot(target.squeeze(1).long(), num_classes=pred.size(1))
    target = target.permute(0, 3, 1, 2).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target - pred * target).sum(dim=(2, 3))
    iou = (intersection + smooth) / (union + smooth)

    B, C, H, W = pred.shape
    y_range = torch.arange(H, device=pred.device).float()
    x_range = torch.arange(W, device=pred.device).float()
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    pred_center_y = (pred * grid_y[None, None]).sum(dim=(2, 3)) / (pred.sum(dim=(2, 3)) + smooth)
    pred_center_x = (pred * grid_x[None, None]).sum(dim=(2, 3)) / (pred.sum(dim=(2, 3)) + smooth)
    gt_center_y = (target * grid_y[None, None]).sum(dim=(2, 3)) / (target.sum(dim=(2, 3)) + smooth)
    gt_center_x = (target * grid_x[None, None]).sum(dim=(2, 3)) / (target.sum(dim=(2, 3)) + smooth)

    dist = ((pred_center_x - gt_center_x)**2 + (pred_center_y - gt_center_y)**2).sqrt()
    s_iou = iou * torch.exp(-0.01 * dist)

    return (1 - s_iou).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):

        B, C, H, W = logits.shape
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=C)
        labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()

        probs = torch.softmax(logits, dim=1)
        ce_loss = -labels_one_hot * torch.log(probs + 1e-8)

        pt = (probs * labels_one_hot).sum(1, keepdim=True)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class DiceLoss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, labels):

        C = logits.shape[1]
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=C)
        labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)

        intersection = (probs * labels_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + labels_one_hot.sum(dim=(0, 2, 3))
        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)
        return dice.mean()

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        laplace_kernel = torch.tensor([[[[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]]]], dtype=torch.float32)
        self.register_buffer("laplace_kernel", laplace_kernel)

    def forward(self, logits, labels):
        B, C, H, W = logits.shape
        probs = torch.softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=C).permute(0,3,1,2).float()

        laplace_kernel = self.laplace_kernel.to(logits.device).repeat(C, 1, 1, 1)

        pred_boundary = F.conv2d(probs, laplace_kernel, padding=1, groups=C).abs()
        gt_boundary = F.conv2d(labels_one_hot, laplace_kernel, padding=1, groups=C).abs()

        loss = F.l1_loss(pred_boundary, gt_boundary)
        return loss

