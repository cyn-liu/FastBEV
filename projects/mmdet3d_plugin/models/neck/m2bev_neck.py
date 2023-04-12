import os
from mmcv.cnn import ConvModule
from torch import nn
import torch.utils.checkpoint as cp

from einops import rearrange

from mmdet3d.registry import MODELS

__all__ = ['M2BevNeck']


class ResModule2D(nn.Module):
    def __init__(self, n_channels, norm_cfg=dict(type='BN2d'), groups=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x


@MODELS.register_module()
class M2BevNeck(nn.Module):
    """Neck for M2BEV.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers=2,
                 norm_cfg=dict(type='BN2d'),
                 stride=2,
                 is_transpose=True,
                 fuse=None,
                 with_cp=False):
        super().__init__()

        self.is_transpose = is_transpose
        self.with_cp = with_cp

        if fuse is not None:
            self.fuse = nn.Conv2d(fuse["in_channels"], fuse["out_channels"], kernel_size=1)
        else:
            self.fuse = None

        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        for i in range(num_layers):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.model = nn.Sequential(*model)

    def forward(self, input_tensor):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (bs, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (bs, C_out, N_y, N_x).
        """

        x = rearrange(input_tensor, 'bs c vx vy vz -> bs (c vz) vx vy')

        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        if self.fuse is not None:
            x = self.fuse(x)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            return [x.transpose(-1, -2)]
        else:
            return [x]
