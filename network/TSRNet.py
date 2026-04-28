import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from tsr.models.transformer.transformer_1d import Transformer1D
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import normal_init, constant_init

from network.xyz_head import XyzRegression

class Triplane(nn.Module):
    def __init__(self, ):
        super(Triplane, self).__init__()

        self.tokenizer = Triplane1DTokenizer()
        self.cross_att = Transformer1D()
        self.xyz_regression = XyzRegression()

    def forward(self, x, y, z, img_feature, cat_id):
        batch_size = img_feature.shape[0]
        img_feature = rearrange(img_feature, 'B C H W-> B (H W) C') # b, 256, 32, 32
        tokens: torch.Tensor = self.tokenizer(batch_size, cat_id) # b ,128, (3 32 32)
        sence_code,  sence_code1= self.cross_att(tokens, encoder_hidden_states = img_feature) # b 128 3*32*32; b 128

        indices_yx = torch.stack((y, x), dim =-1) # b, 1, 64, 64, 2
        indices_zx = torch.stack((z, x), dim =-1) # b, 1, 64, 64, 2
        indices_zy = torch.stack((z, y), dim =-1) # b, 1, 64, 64, 2

        indices = torch.cat([indices_yx, indices_zx, indices_zy],dim=1) # B ,3, H, W ,2
        indices = rearrange(indices, "B Np Hp Wp Ip-> (B Np) Hp Wp Ip")
        indices = torch.clamp(indices, min=-0.5, max=0.5) * 2

        sence_code = rearrange(sence_code, "B Cp (Np Hp Wp) -> (B Np) Cp Hp Wp", Np=3, Hp=32, Wp=32)

        var_feat = F.grid_sample(sence_code, indices, align_corners=False, mode= "bilinear" )
        var_feat = rearrange(var_feat, "(B Np) Cp Hp Wp -> B (Np Cp) Hp Wp ", Np=3) # b, 374, h, w

        plane_xyz  = rearrange(sence_code, "(B Np) Cp Hp Wp -> B Cp Np Hp Wp", Np=3, Hp=32, Wp=32)
        # plane_xyz = rearrange(sence_code0, "B Cp (Np Hp Wp) -> B Cp Np Hp Wp", Np=3, Hp=32, Wp=32)

        plane_xy = plane_xyz[:, :, 0, None, :, :].repeat(1, 1, 32, 1, 1)
        plane_zx = plane_xyz[:, :, 1, :, None, :].repeat(1, 1, 1, 32, 1)
        plane_zy = plane_xyz[:, :, 2, :, :, None].repeat(1, 1, 1, 1, 32)
        occ_feat = torch.cat([plane_xy, plane_zx, plane_zy], dim=1)

        plane_xyz = rearrange(sence_code1, "B Cp (Np Hp Wp) -> B Cp Np Hp Wp", Np=3, Hp=32, Wp=32)

        plane_xy = plane_xyz[:, :, 0, None, :, :].repeat(1, 1, 32, 1, 1)
        plane_zx = plane_xyz[:, :, 1, :, None, :].repeat(1, 1, 1, 32, 1)
        plane_zy = plane_xyz[:, :, 2, :, :, None].repeat(1, 1, 1, 1, 32)
        occ_feat1 = torch.cat([plane_xy, plane_zx, plane_zy], dim=1)


        return occ_feat, occ_feat1, var_feat


class Triplane1DTokenizer(nn.Module):
    def __init__(self, ):
        super(Triplane1DTokenizer, self).__init__()
        self.embeddings = nn.Parameter(
            torch.randn(
                (6, 3, 128, 32, 32),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(128)
        )

    def forward(self, batch_size: int, cat_id) -> torch.Tensor:

        # return rearrange(
        #     repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size),
        #     "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        # )
        return rearrange(
            torch.index_select(self.embeddings, 0, cat_id),
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

class OccNet(nn.Module):
    def __init__(self, ):
        super(OccNet, self).__init__()

        self.conv1 = nn.Conv1d(384, 128, 1)
        self.conv2 = nn.Conv1d(128, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

    def forward(self, x):

        x = x.flatten(2)
        x = F.relu((self.conv1(x)))
        x = self.conv2(x)

        return x.squeeze(1)


