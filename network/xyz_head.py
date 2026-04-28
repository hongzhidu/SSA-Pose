import ipdb
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
# from mmcv.cnn import normal_init, constant_init
from mmengine.model import normal_init, constant_init
# from timm.models.layers import StdConv2d
from network.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from network.torch_utils.layers.conv_module import ConvModule
from network.torch_utils.layers.std_conv_transpose import StdConvTranspose2d


class TopDownMaskXyzHead(nn.Module):
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

class XyzRegression(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv_out = nn.Conv2d(256, 3, kernel_size=1, padding=(1 - 1) // 2, bias=True, )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def forward(self, x):

        x = self.conv_out(x)

        coor_x = x[:, 0:1, :, :]
        coor_y = x[:, 1:2, :, :]
        coor_z = x[:, 2:3, :, :]


        return coor_x, coor_y, coor_z


class Upsample_regression(nn.Module):
    def __init__(
        self,
        in_dim,
        norm="GN",
        num_gn_groups=32,
        act="GELU",
    ):
        super().__init__()

        self.deconv_0 =  nn.ConvTranspose2d(
                        in_dim,
                        256,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    )
        self.norm_0 = get_norm("GN", 256, num_gn_groups=32)
        self.act_0 = get_nn_act_func("GELU")

        self.conv_0a = ConvModule(
            256,
            256,
            kernel_size=3,
            padding=(3 - 1) // 2,
            conv_cfg= None,
            norm=norm,
            num_gn_groups=num_gn_groups,
            act=act,
        )
        self.conv_0b = ConvModule(
            256,
            256,
            kernel_size=3,
            padding=(3 - 1) // 2,
            conv_cfg= None,
            norm=norm,
            num_gn_groups=num_gn_groups,
            act=act,
        )

        self.deconv_1 =  nn.ConvTranspose2d(
                        256,
                        256,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    )
        self.norm_1 = get_norm("GN", 256, num_gn_groups=32)
        self.act_1 = get_nn_act_func("GELU")

        self.conv_1a = ConvModule(
            256,
            256,
            kernel_size=3,
            padding=(3 - 1) // 2,
            conv_cfg= None,
            norm=norm,
            num_gn_groups=num_gn_groups,
            act=act,
        )
        self.conv_1b = ConvModule(
            256,
            256,
            kernel_size=3,
            padding=(3 - 1) // 2,
            conv_cfg= None,
            norm=norm,
            num_gn_groups=num_gn_groups,
            act=act,
        )
        self.deconv_2 =  nn.ConvTranspose2d(
                        256,
                        256,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    )
        self.norm_2 = get_norm("GN", 256, num_gn_groups=32)
        self.act_2 = get_nn_act_func("GELU")

        self.conv_2a = ConvModule(
            256,
            256,
            kernel_size=3,
            padding=(3 - 1) // 2,
            conv_cfg= None,
            norm=norm,
            num_gn_groups=num_gn_groups,
            act=act,
        )
        self.conv_2b = ConvModule(
            256,
            256,
            kernel_size=3,
            padding=(3 - 1) // 2,
            conv_cfg= None,
            norm=norm,
            num_gn_groups=num_gn_groups,
            act=act,
        )

        self.conv_out = nn.Conv2d(256, 3, kernel_size=1, padding=(1 - 1) // 2, bias=True, )

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

        x0 = self.deconv_0(x)
        x0 = self.norm_0(x0)
        x0 = self.act_0(x0)
        x0 = self.conv_0a(x0)
        x0 = self.conv_0b(x0)

        x1 = self.deconv_1(x0)
        x1 = self.norm_1(x1)
        x1 = self.act_1(x1)
        x1 = self.conv_1a(x1)
        x1 = self.conv_1b(x1)

        x2 = self.deconv_2(x1)
        x2 = self.norm_2(x2)
        x2 = self.act_2(x2)
        x2 = self.conv_2a(x2)
        x2 = self.conv_2b(x2)

        x = self.conv_out(x2)

        coor_x = x[:, 0:1, :, :]
        coor_y = x[:, 1:2, :, :]
        coor_z = x[:, 2:3, :, :]


        return coor_x, coor_y, coor_z, x1, x2




if __name__ == '__main__':
    xyz_head = TopDownMaskXyzHead(in_dim=1024)
