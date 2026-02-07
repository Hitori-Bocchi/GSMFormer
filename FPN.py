# 自适应的Agent Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from typing import Dict, List, Optional, Callable, Any
from settings import TeLU

Tensor = torch.Tensor
ExtraFPNBlock = Any


class AgentAttention(nn.Module):

    def __init__(self, channels: int, agent_num: int = 49, agent_dim: int = 32):
        super().__init__()
        self.channels = channels
        self.agent_num = agent_num

        pool_size = int(agent_num ** 0.5)
        self.pool_size = (pool_size, pool_size)

        self.agent_dim = agent_dim
        self.scale = agent_dim ** -0.5

        def build_proj(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.Conv2d(out_c, out_c, kernel_size=(3,1), padding=(1,0), groups=out_c, bias=False),
                nn.Conv2d(out_c, out_c, kernel_size=(1,3), padding=(0,1), groups=out_c, bias=False),
                nn.BatchNorm2d(out_c),
                TeLU(),
                nn.Dropout2d(0.1)
            )

        self.to_q = build_proj(channels, agent_dim)
        self.to_k = build_proj(channels, agent_dim)
        self.to_v = build_proj(channels, channels)
        self.to_a = build_proj(channels, agent_dim)

        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
        self.out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, lateral: Tensor, topdown: Tensor) -> Tensor:

        B, C, H, W = lateral.shape
        q = self.to_q(lateral).flatten(2).transpose(1, 2)
        k = self.to_k(topdown).flatten(2)
        v = self.to_v(topdown).flatten(2).transpose(1, 2)
        a_feat = self.to_a(topdown)
        a = self.pool(a_feat).flatten(2).transpose(1, 2)
        attn_KA = (a @ k) * self.scale
        attn_KA = torch.softmax(attn_KA, dim=-1)
        agent_v = attn_KA @ v
        attn_QA = (q @ a.transpose(1, 2)) * self.scale
        attn_QA = torch.softmax(attn_QA, dim=-1)

        global_context = attn_QA @ agent_v

        global_context = rearrange(global_context, 'b (h w) c -> b c h w', h=H, w=W)
        out = self.out(global_context)

        return lateral + out

class FeaturePyramidNetwork(nn.Module):

    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks: Optional[ExtraFPNBlock] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            agent_dim: int = 32,
            agent_num: int = 49
    ):
        super().__init__()
        self.extra_blocks = extra_blocks

        num_layers = min(4, len(in_channels_list))
        self.num_layers = num_layers
        in_channels_list = in_channels_list[:num_layers]
        self.out_channels = out_channels

        self.agent_fusions = nn.ModuleList([
            AgentAttention(out_channels, agent_num=agent_num, agent_dim=agent_dim)
            for _ in range(num_layers)
        ])

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.gates = nn.ModuleList()

        for in_ch in in_channels_list:

            inner = nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=True)
            layer = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=True),
            )

            self.inner_blocks.append(inner)
            self.layer_blocks.append(layer)

            gate = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=True)
            nn.init.constant_(gate.weight, 0.01)
            nn.init.constant_(gate.bias, -5.0)
            self.gates.append(gate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in self.gates:
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        return self.inner_blocks[idx](x)

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        return self.layer_blocks[idx](x)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:

        names = list(x.keys())[: self.num_layers]
        values = list(x.values())[: self.num_layers]

        last_inner = self.get_result_from_inner_blocks(values[-1], -1)
        results = [self.get_result_from_layer_blocks(last_inner, -1)]

        for idx in range(len(values) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(values[idx], idx)
            feat_shape = inner_lateral.shape[-2:]

            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")

            fused_inner = self.agent_fusions[idx](inner_lateral, inner_top_down)

            gate_in = torch.cat([inner_lateral, inner_top_down], dim=1)
            gate = torch.sigmoid(self.gates[idx](gate_in))

            last_inner = fused_inner * gate + inner_top_down * (1.0 - gate)

            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, values, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2
    in_channels_list = [64, 128, 256, 512]
    out_channels = 256
    sizes = [(128, 128), (64, 64), (32, 32), (16, 16)]

    features = OrderedDict()
    for i, (ch, size) in enumerate(zip(in_channels_list, sizes)):
        H, W = size
        features[f"p{i}"] = torch.randn(B, ch, H, W, device=device)

    fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, agent_num=49)
    fpn.to(device)
    fpn.eval()

    with torch.no_grad():
        out = fpn(features)

    print("FPN outputs shapes:")
    for k, v in out.items():
        print(f"{k}: {v.shape}")