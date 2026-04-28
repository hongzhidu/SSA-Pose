import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import normal_init, constant_init
from network.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from torchvision.transforms import Resize, InterpolationMode

import absl.flags as flags
FLAGS = flags.FLAGS

class PoseEstimator(nn.Module):
    def __init__(self,
        featdim=128,
        rot_dim=6,
        num_extra_layers=0,
        norm="GN",
        num_gn_groups=32,
        act="relu",
        flat_op="flatten",
        final_spatial_size=(8, 8),
    ):
        super().__init__()
        self.featdim = featdim
        self.flat_op = flat_op

        conv_act = get_nn_act_func(act)
        self.act = get_nn_act_func("lrelu")  # legacy model

        conv_layer =  nn.Conv2d
        self.features = nn.ModuleList()


        self.features.append(
            conv_layer(
                3+2+1+256+384,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        self.features.append(get_norm(norm, 256, num_gn_groups=num_gn_groups))
        self.features.append(conv_act)
        self.features.append(
            conv_layer(
                256,
                featdim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
        self.features.append(conv_act)
        self.features.append(
            conv_layer(
                featdim,
                featdim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
        self.features.append(conv_act)

        for i in range(num_extra_layers):
            self.features.append(
                conv_layer(
                    featdim,
                    featdim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        final_h, final_w = final_spatial_size
        fc_in_dim = {
            "flatten": featdim * final_h * final_w,
            "avg": featdim,
            "avg-max": featdim * 2,
            "avg-max-min": featdim * 3,
        }[flat_op]

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, rot_dim)  # quat or rot6d
        self.fc_t = nn.Linear(256, 2)

        self.fc1_z = nn.Linear(fc_in_dim, 1024)
        self.fc2_z = nn.Linear(1024, 256)
        self.fc_z = nn.Linear(256, 1)

        self.resize_func_input = Resize(FLAGS.out_res, interpolation=InterpolationMode.NEAREST)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)


    def forward(self, coor_feat, fused_feature, mask_attention):


        x = coor_feat

        x = torch.cat([x, mask_attention, fused_feature], dim=1)

        for _i, layer in enumerate(self.features):
            x = layer(x)

        flat_conv_feat = x.flatten(2)  # [B,featdim,*]
        flat_conv_feat = flat_conv_feat.flatten(1)

        x = self.act(self.fc1(flat_conv_feat))
        x = self.act(self.fc2(x))
        #
        rot = self.fc_r(x)
        t = self.fc_t(x)

        xz = self.act(self.fc1_z(flat_conv_feat))
        xz = self.act(self.fc2_z(xz))
        z = self.fc_z(xz)

        t = torch.cat([t, z], dim=1)

        return rot, t

 