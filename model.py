
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.ops import deform_conv2d
from FPN import FeaturePyramidNetwork
from settings import TeLU

try:
    _ = timm.create_model
except Exception:
    pass


class StructureGuidedDynamicCDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        geom_feat_dim = 4

        self.offset_conv = nn.Conv2d(geom_feat_dim, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, bias=True)

        self.mask_conv = nn.Conv2d(geom_feat_dim, kernel_size * kernel_size, kernel_size=3, padding=1, bias=True)

        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        x_geom = x.mean(dim=1, keepdim=True)

        x_pad = F.pad(x_geom, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(x_pad, self.sobel_x)
        gy = F.conv2d(x_pad, self.sobel_y)

        Ixx = gx ** 2
        Iyy = gy ** 2
        Ixy = gx * gy

        tr = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy

        delta = torch.sqrt((tr * tr - 4 * det).clamp(min=1e-6))

        l1 = (tr + delta) / 2
        l2 = (tr - delta) / 2

        geom_invariants = torch.cat([l1, l2, tr, det], dim=1)

        offsets = self.offset_conv(geom_invariants)
        masks = torch.sigmoid(self.mask_conv(geom_invariants))

        out = deform_conv2d(input=x,
                            offset=offsets,
                            weight=self.weight,
                            bias=self.bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=1,
                            mask=masks)

        return out

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super().__init__()

        self.hidden_dim = max(32, out_channels * 4)

        self.cdc_spatial = StructureGuidedDynamicCDC(in_channels, self.hidden_dim,
                                                     kernel_size, stride,
                                                     padding, dilation, groups, bias)

        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.act1 = TeLU()

        self.mlp = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            TeLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            TeLU()
        )

        self.proj_out = nn.Conv2d(self.hidden_dim, out_channels, kernel_size=1, bias=True)
        self.aux_loss = None

    def forward(self, x):
        x_grad = self.act1(self.bn1(self.cdc_spatial(x)))
        x_refined = x_grad + self.mlp(x_grad)
        out = self.proj_out(x_refined)
        return out

class AdaptiveCornerTargetGen(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.act = TeLU()

    def forward(self, gt_mask):

        img = gt_mask.float()

        gx = F.conv2d(img, self.sobel_x, padding=1)
        gy = F.conv2d(img, self.sobel_y, padding=1)

        Ixx = gx * gx
        Iyy = gy * gy
        Ixy = gx * gy

        det = Ixx * Iyy - Ixy ** 2
        tr = Ixx + Iyy
        k = 0.04
        response = det - k * (tr ** 2)

        response = F.relu(response)

        B = response.shape[0]
        view_flat = response.view(B, -1)
        max_val = view_flat.max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        response = response / (max_val + 1e-6)


        return response.repeat(1, 8, 1, 1)


class CDCWeightedL1(nn.Module):
    def __init__(self, pos_weight=30.0, eps=1e-6):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.eps = float(eps)

    def forward(self, pred, target):
        assert pred.shape == target.shape, "pred & target must match shape for aux loss"
        weights = 1.0 + target * self.pos_weight
        diff = torch.abs(pred - target)
        weighted = diff * weights
        loss = weighted.sum() / (weights.sum() + self.eps)
        return loss


class SwinWrapper(nn.Module):
    def __init__(self, timm_model, geom_proj: nn.Module = None, final_fuse: nn.Module = None):
        super().__init__()
        self.model = timm_model
        if geom_proj is not None:
            self.geom_proj = geom_proj
        if final_fuse is not None:
            self.final_fuse = final_fuse

    def forward(self, x):
        return self.model(x)

    @property
    def feature_info(self):
        return self.model.feature_info


class SwinBackboneWithAdaptiveGeomFPN(nn.Module):
    def __init__(self,
                 img_size=512,
                 out_channels=256,
                 swin_variant='swin_small_patch4_window7_224',
                 pretrained=True,
                 aux_loss_weight=1.0,
                 detach_swin_for_fusion=True,
                 cdc_theta=1.0,
                 aux_pos_weight=30.0):
        super().__init__()
        swin_kwargs = {"pretrained": pretrained, "features_only": True, "img_size": img_size}
        try:
            timm_swin = timm.create_model(swin_variant, **swin_kwargs)
        except TypeError:
            swin_kwargs.pop("img_size", None)
            timm_swin = timm.create_model(swin_variant, **swin_kwargs)

        in_channels_list = timm_swin.feature_info.channels()
        if not isinstance(in_channels_list, (list, tuple)) or len(in_channels_list) == 0:
            raise RuntimeError("swin.feature_info.channels() returned unexpected value")

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        self.cdc = Conv2d_cd(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False, theta=cdc_theta)

        self.cdc_target_gen = AdaptiveCornerTargetGen()

        swin_stage0_ch = self.in_channels_list[0]

        geom_proj = nn.Conv2d(in_channels=8, out_channels=swin_stage0_ch, kernel_size=1, bias=True)
        nn.init.constant_(geom_proj.bias, 0.0)

        final_fuse = nn.Sequential(
            nn.Conv2d(swin_stage0_ch, swin_stage0_ch, kernel_size=(3,1), padding=(1,0), bias=False),
            nn.Conv2d(swin_stage0_ch, swin_stage0_ch, kernel_size=(1,3), padding=(0,1), bias=False),
            nn.BatchNorm2d(swin_stage0_ch),
            TeLU(),
            nn.Dropout2d(0.1),
        )

        self.swin = SwinWrapper(timm_swin, geom_proj=geom_proj, final_fuse=final_fuse)

        self.post_fuse = nn.Sequential(
            nn.Conv2d(swin_stage0_ch, swin_stage0_ch, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.Conv2d(swin_stage0_ch, swin_stage0_ch, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(swin_stage0_ch),
            TeLU(),
            nn.Dropout2d(0.1)
        )

        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)
        self.pointwise = nn.Conv2d(in_channels=self.in_channels_list[-1],
                                   out_channels=self.out_channels,
                                   kernel_size=1,
                                   bias=False)

        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_criterion = CDCWeightedL1(pos_weight=aux_pos_weight)
        self.aux_loss = None

        self.detach_swin_for_fusion = bool(detach_swin_for_fusion)

    @property
    def geom_proj(self):
        return getattr(self.swin, "geom_proj", None)

    @property
    def final_fuse(self):
        return getattr(self.swin, "final_fuse", None)

    @staticmethod
    def _ensure_bchw(feat, expected_ch=None, name=None):
        if feat.dim() != 4:
            raise RuntimeError(f"feature {name} expected 4D tensor, got {tuple(feat.shape)}")
        if expected_ch is not None:
            if feat.shape[1] == expected_ch:
                return feat
            if feat.shape[-1] == expected_ch:
                return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x, gt=None):
        self.aux_loss = None

        if isinstance(x, (list, tuple)):
            try:
                x = torch.stack(x, dim=0)
            except Exception:
                x = x[0]

        swin_feats = self.swin(x)
        if not isinstance(swin_feats, (list, tuple)) or len(swin_feats) == 0:
            raise RuntimeError("swin returned unexpected features")

        swin_feats_bchw = []
        for i, f in enumerate(swin_feats):
            exp_ch = self.in_channels_list[i] if i < len(self.in_channels_list) else None
            f2 = self._ensure_bchw(f, expected_ch=exp_ch, name=f"swin_feat_{i}")
            swin_feats_bchw.append(f2)

        swin0 = swin_feats_bchw[0]
        B, Cs, H, W = swin0.shape

        if x.shape[1] == 3:
            img_gray = x.mean(dim=1, keepdim=True)
        else:
            img_gray = x
        if img_gray.shape[2:] != (H, W):
            img_gray_resized = F.interpolate(img_gray, size=(H, W), mode='bilinear', align_corners=False)
        else:
            img_gray_resized = img_gray

        cdc8 = self.cdc(img_gray_resized)

        if self.training and gt is not None:

            if gt.dim() == 4 and gt.shape[1] != 1:
                gt_gray = gt.mean(dim=1, keepdim=True)
            elif gt.dim() == 3:
                gt_gray = gt.unsqueeze(1)
            else:
                gt_gray = gt

            if not torch.is_floating_point(gt_gray):
                gt_gray = gt_gray.to(dtype=img_gray_resized.dtype)

            if gt_gray.shape[2:] != (H, W):
                gt_gray_resized = F.interpolate(gt_gray, size=(H, W), mode='bilinear', align_corners=False)
            else:
                gt_gray_resized = gt_gray


            with torch.no_grad():
                target = self.cdc_target_gen(gt_gray_resized)
                target = target.to(dtype=cdc8.dtype, device=cdc8.device)


            self.aux_loss = self.aux_criterion(cdc8, target) * self.aux_loss_weight
        else:
            self.aux_loss = None

        geom_proj = self.geom_proj

        gate = torch.sigmoid(geom_proj(cdc8))

        if self.detach_swin_for_fusion:
            swin0_for_fuse = swin0.detach()
        else:
            swin0_for_fuse = swin0

        fused0 = swin0 + swin0_for_fuse * gate

        # refine
        fused0 = self.final_fuse(fused0)
        fused0 = self.post_fuse(fused0)

        out = {}
        for i, feat in enumerate(swin_feats_bchw):
            out[str(i)] = fused0 if i == 0 else feat

        fpn_outs = self.fpn(out)

        low_level_feat = swin_feats_bchw[-1]
        low_bchw = low_level_feat
        if low_level_feat.dim() != 4 or low_level_feat.shape[1] != self.in_channels_list[-1]:
            try:
                low_bchw = low_level_feat.permute(0, 3, 1, 2).contiguous()
            except Exception:
                raise RuntimeError("low_level_feat unexpected shape, can't convert to (B,C,H,W)")

        low_bchw = self.pointwise(low_bchw)
        return fpn_outs, low_bchw


class hierarchical_attention_head(nn.Module):
    def __init__(self, in_channels, low_in_ch, num_classes, mid_channels=None, init_gate_bias=0):
        super().__init__()
        if mid_channels is None:
            mid_channels = max(128, in_channels // 2)

        self.body_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        )
        self.body_bn1 = nn.BatchNorm2d(mid_channels)
        self.body_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        )
        self.body_bn2 = nn.BatchNorm2d(mid_channels)
        self.body_act = TeLU()

        self.boundary_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        )
        self.boundary_bn1 = nn.BatchNorm2d(mid_channels)
        self.boundary_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        )
        self.boundary_bn2 = nn.BatchNorm2d(mid_channels)
        self.boundary_act = TeLU()

        self.branch1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, (3, 1), padding=(1, 0), bias=False),
            nn.Conv2d(mid_channels, mid_channels, (1, 3), padding=(0, 1), bias=False)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, (5, 1), padding=(2, 0), bias=False),
            nn.Conv2d(mid_channels, mid_channels, (1, 5), padding=(0, 2), bias=False)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, (7, 1), padding=(3, 0), bias=False),
            nn.Conv2d(mid_channels, mid_channels, (1, 7), padding=(0, 3), bias=False)
        )

        self.ms_bn = nn.BatchNorm2d(mid_channels * 3)
        self.ms_act = TeLU()
        self.ms_pointwise = nn.Conv2d(mid_channels * 3, mid_channels, 1, bias=False)

        self.gate_conv = nn.Conv2d(mid_channels * 2, 1, 1)
        nn.init.constant_(self.gate_conv.bias, init_gate_bias)

        self.dropout = nn.Dropout2d(0.1)

        self.out_conv = nn.Conv2d(mid_channels, 2, 1)  # will be replaced in wrapper
        self.edge_out = nn.Conv2d(mid_channels, 1, 1)

    def forward(self, top_feat, low_feat, out_size):
        body = self.body_act(self.body_bn1(self.body_conv1(top_feat)))
        body = self.body_act(self.body_bn2(self.body_conv2(body)))

        up_low = F.interpolate(low_feat, size=body.shape[2:], mode='bilinear', align_corners=False)
        boundary = top_feat - up_low

        edge = self.boundary_act(self.boundary_bn1(self.boundary_conv1(boundary)))
        edge = self.boundary_act(self.boundary_bn2(self.boundary_conv2(edge)))

        gate = torch.sigmoid(self.gate_conv(torch.cat([body, edge], dim=1)))
        fused = body + gate * (edge - body)

        x = fused

        feat1 = self.branch1(fused)
        feat2 = self.branch2(fused)
        feat3 = self.branch3(fused)
        ms_feats = torch.cat([feat1, feat2, feat3], dim=1)
        ms_feats = self.ms_bn(ms_feats)
        ms_feats = self.ms_act(ms_feats)
        ms_feats = self.dropout(ms_feats)
        fused = self.ms_pointwise(ms_feats)
        fused = fused + x

        out = self.out_conv(fused)
        out = F.interpolate(out, size=out_size, mode='bilinear', align_corners=False)
        return out


class GlobalSegWrapperHDNet(nn.Module):
    def __init__(self, backbone, num_classes=2, mid_channels=None):
        super().__init__()
        self.backbone = backbone
        top_in_ch = getattr(backbone, "out_channels", 256)
        low_in_ch = backbone.in_channels_list[-1]
        self.head = hierarchical_attention_head(in_channels=top_in_ch,
                                               low_in_ch=low_in_ch,
                                               num_classes=num_classes,
                                               mid_channels=mid_channels,
                                               init_gate_bias=-6)
        self.head.out_conv = nn.Conv2d(max(128, top_in_ch // 2), num_classes, 1)

    def forward(self, images, gt=None):
        if isinstance(images, (list, tuple)):
            imgs = torch.stack(images, dim=0)
        else:
            imgs = images

        fpn_outs, low_level_feat = self.backbone(imgs, gt=gt)
        aux_loss = getattr(self.backbone, "aux_loss", None)

        keys = sorted(fpn_outs.keys())
        top_feat = fpn_outs[keys[0]]

        H, W = imgs.shape[2], imgs.shape[3]
        logits = self.head(top_feat, low_level_feat, out_size=(H, W))

        return logits, aux_loss


def get_model(num_classes=2, img_size=512, swin_variant='swin_tiny_patch4_window7_224', pretrained=True,
              aux_loss_weight=1.0, aux_pos_weight=30.0):
    backbone = SwinBackboneWithAdaptiveGeomFPN(img_size=img_size, out_channels=256,
                                               swin_variant=swin_variant, pretrained=pretrained,
                                               aux_loss_weight=aux_loss_weight, aux_pos_weight=aux_pos_weight)
    backbone.out_channels = 256
    model = GlobalSegWrapperHDNet(backbone, num_classes=num_classes, mid_channels=None)
    return model

