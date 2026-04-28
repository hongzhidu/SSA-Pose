import numpy as np

import torch
import torch.nn as nn

from torch.nn import init
from network.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import normal_init, constant_init

# from timm.models.layers import StdConv2d
from network.torch_utils.layers.conv_module import ConvModule
from network.torch_utils.layers.std_conv_transpose import StdConvTranspose2d
import absl.flags as flags
FLAGS = flags.FLAGS


class AttentionMaskHead(nn.Module):
    def __init__(self, conv_layer=nn.Conv2d, act='relu', featdim=128, use_rgb_feature=False, add_dim=0, out_dim=1):
        super(AttentionMaskHead, self).__init__()

        self.act = get_nn_act_func(act)


        self.conv1 = conv_layer(
                3 + 2,
                featdim,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        self.norm1 = get_norm("GN", featdim, num_gn_groups=32)

        self.conv2 = conv_layer(
            128  + 256 + 384,
                featdim,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        self.norm2 = get_norm("GN", featdim, num_gn_groups=32)
        self.conv3 = conv_layer(
                featdim,
                featdim,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        self.norm3 = get_norm("GN", featdim, num_gn_groups=32)
        self.conv4 = conv_layer(
            featdim,
            out_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, coor_feat, mask, rgb_feat=None):
        x = self.conv1(coor_feat)
        x = self.norm1(x)
        x = self.act(x)

        x = torch.cat([x, rgb_feat], dim=1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv4(x)
        log_var = x.clone()
        log_var = torch.clip(log_var, min=FLAGS.log_var_min)
        att_mask = 1 / (torch.exp(log_var) + 1e-5)
        # VALUE RANGE [0,3]
        att_mask = att_mask.clip(0, 5)
        att_mask[att_mask < FLAGS.att_thr] = 0

        return att_mask, log_var


class UpsampleHead(nn.Module):
    def __init__(
        self,
        in_dim,
        up_types=("deconv", "bilinear", "bilinear"),
        deconv_kernel_size=3,
        num_conv_per_block=2,
        feat_dim=256,
        feat_kernel_size=3,
        use_ws=False,
        use_ws_deconv=False,
        norm="GN",
        num_gn_groups=32,
        act="GELU",
        out_kernel_size=1,
    ):
        """
        Args:
            up_types: use up-conv or deconv for each up-sampling layer
                ("bilinear", "bilinear", "bilinear")
                ("deconv", "bilinear", "bilinear")  # CDPNv2 rot head
                ("deconv", "deconv", "deconv")  # CDPNv1 rot head
                ("nearest", "nearest", "nearest")  # implement here but maybe won't use
        NOTE: default from stride 32 to stride 4 (3 ups)
        """
        super().__init__()
        assert out_kernel_size in [
            1,
            3,
        ], "Only support output kernel size: 1 and 3"
        assert deconv_kernel_size in [
            1,
            3,
            4,
        ], "Only support deconv kernel size: 1, 3, and 4"
        assert len(up_types) > 0, up_types

        self.features = nn.ModuleList()

        for i, up_type in enumerate(up_types):
            _in_dim = in_dim if i == 0 else feat_dim
            if up_type == "deconv":
                (
                    deconv_kernel,
                    deconv_pad,
                    deconv_out_pad,
                ) = _get_deconv_pad_outpad(deconv_kernel_size)
                deconv_layer = StdConvTranspose2d if use_ws_deconv else nn.ConvTranspose2d
                self.features.append(
                    deconv_layer(
                        _in_dim,
                        feat_dim,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=deconv_pad,
                        output_padding=deconv_out_pad,
                        bias=False,
                    )
                )
                self.features.append(get_norm(norm, feat_dim, num_gn_groups=num_gn_groups))
                self.features.append(get_nn_act_func(act))
            elif up_type == "bilinear":
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
            elif up_type == "nearest":
                self.features.append(nn.UpsamplingNearest2d(scale_factor=2))
            else:
                raise ValueError(f"Unknown up_type: {up_type}")

            if up_type in ["bilinear", "nearest"]:
                assert num_conv_per_block >= 1, num_conv_per_block
            for i_conv in range(num_conv_per_block):
                if i == 0 and i_conv == 0 and up_type in ["bilinear", "nearest"]:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = feat_dim

                if use_ws:
                    conv_cfg = dict(type="StdConv2d")
                else:
                    conv_cfg = None

                self.features.append(
                    ConvModule(
                        conv_in_dim,
                        feat_dim,
                        kernel_size=feat_kernel_size,
                        padding=(feat_kernel_size - 1) // 2,
                        conv_cfg=conv_cfg,
                        norm=norm,
                        num_gn_groups=num_gn_groups,
                        act=act,
                    )
                )


        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)


    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        for i_layer, l in enumerate(self.features):
            x = l(x)
        return x


def _get_deconv_pad_outpad(deconv_kernel):
    """Get padding and out padding for deconv layers."""
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0
    else:
        raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

    return deconv_kernel, padding, output_padding