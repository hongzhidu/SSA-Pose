import torch
import torch.nn as nn
import torch.nn.functional as F
import absl.flags as flags
from absl import app
import mmcv
FLAGS = flags.FLAGS  # can control the weight of each term here
from tools.training_utils import get_gt_v
from tools.rot_utils import get_rot_vec_vert_batch, get_rot_mat_y_first, get_vertical_rot_vec
import numpy as np

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()
        if FLAGS.pose_loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='none')
        elif FLAGS.pose_loss_type == 'smoothl1':   # same as MSE
            self.loss_func = nn.SmoothL1Loss(beta=0.5, reduction='none')
        else:
            raise NotImplementedError
        self.use_pnp = FLAGS.use_pnp > 0
        self.threshold = 0.03

    def forward(self, pred_dict, data):
        loss_dict = {}
        device = pred_dict["rot"].device
        bs = pred_dict["rot"].shape[0]
        gt_rotation= data['rotation'].to(device)
        gt_translation = data["translation"].to(device)
        gt_size = data["real_size"].to(device)
        gt_mask = data["roi_mask_output"].to(device)
        sym = data['sym_info'].to(device)
        nocs_scale = data['nocs_scale'].to(device).unsqueeze(-1)

        gt_size_norm = gt_size / nocs_scale
        gt_translation_norm = gt_translation / nocs_scale
        sym_mask = sym[:, 0] == 1
        gt_coor = data['nocs_coord'].to(device)


        loss_dict["Rot1"] = FLAGS.rot_1_w * self.cal_loss_Rot1(pred_dict["rot"], gt_rotation)
        loss_dict["Tran"] = FLAGS.tran_w * self.cal_loss_tran_scale(pred_dict["trans"], gt_translation_norm)
        loss_dict["Size"] = FLAGS.size_w * self.cal_loss_tran_scale(pred_dict["size"], gt_size_norm)
        model_point_norm = data['model_point'].to(device)
        loss_dict["Point_matching"] = FLAGS.prop_pm_w * self.point_matching_loss(model_point_norm,
                                                                          pred_dict['rot'],
                                                                          pred_dict["trans"],
                                                                          gt_rotation,
                                                                          gt_translation_norm)

        loss_dict["coor"] = FLAGS.coor_w * self.cal_coor_loss(pred_dict["coor"], gt_coor, gt_mask, sym,  pred_dict["log_var"], sym_mask)
        loss_dict["occ"] = F.binary_cross_entropy_with_logits(pred_dict["pred_occ"], data["occ_partial"].to(device))
        loss_dict["occ1"] = F.binary_cross_entropy_with_logits(pred_dict["pred_occ1"], data["occ_complete"].to(device))

        return loss_dict


    def cal_loss_consistency(self, nocs, pose, mask, data):
        device = nocs.device
        nocs_resize = torch.flatten(nocs, -2, -1)
        nocs_4dim = torch.cat(
            [nocs_resize, torch.ones([nocs_resize.shape[0], 1, nocs_resize.shape[2]], device=device)],
            dim=1)
        roi_coor_2d = data['roi_coord_2d'].to(device)
        roi_cams = data['cam_K'].to(device)
        pose_nocs = torch.bmm(pose, nocs_4dim)
        pose_nocs = pose_nocs[:, :3]
        reproj_nocs = torch.bmm(roi_cams, pose_nocs)
        reproj_nocs = reproj_nocs.reshape(nocs.shape)
        reproj_2d = reproj_nocs[:, :2] / reproj_nocs[:, 2, None]
        delta_2d = torch.norm(reproj_2d - roi_coor_2d, dim=1)
        roi_mask = mask
        delta_2d = roi_mask.squeeze(1) * delta_2d
        delta_2d = delta_2d.clamp(max=FLAGS.con_threshold)
        delta_2d_mean = delta_2d.sum(dim=[1, 2]) / roi_mask.sum(dim=[1, 2, 3])
        return delta_2d_mean.mean()

    def cal_coor_loss(self, pred_coor, gt_coor, gt_mask, sym, log_var, sym_mask):
        # filter out invalid point
        pred_coor = pred_coor * gt_mask
        gt_coor = gt_coor * gt_mask

        return self.cal_coor_loss_for_batch(pred_coor, gt_coor, gt_mask, log_var, sym_mask)

    def cal_coor_loss_for_each_item(self, coords, nocs, gt_mask, log_variance=None):
        if FLAGS.use_uncertainty_loss:
            corr_loss = 15 * torch.exp(-0.5 * log_variance) * torch.abs(coords - nocs).sum(0).unsqueeze(0) + 0.5 * log_variance
            corr_loss = torch.sum(corr_loss * gt_mask) / (torch.sum(gt_mask) + 1e-5)
        else:
            diff = torch.abs(coords - nocs)
            lower_corr_loss = torch.pow(diff, 2) / (2.0 * self.threshold)
            higher_corr_loss = diff - self.threshold / 2.0
            corr_loss_matrix = torch.where(diff > self.threshold, higher_corr_loss, lower_corr_loss)
            corr_loss_matrix = gt_mask * corr_loss_matrix
            corr_loss = torch.sum(corr_loss_matrix) / (torch.sum(gt_mask) + 1e-5)
        return corr_loss

    def cal_coor_loss_for_batch(self, coords, nocs, gt_mask, log_variance, sym_mask):

        diff = torch.abs(coords - nocs)

        assert log_variance.shape[1] == FLAGS.mask_dim
        if FLAGS.use_uncertainty_loss and FLAGS.uncertain_type == 'la':
            if FLAGS.mask_dim == 1:
                corr_loss = 15 * torch.exp(-0.5 * log_variance) * diff.sum(1).unsqueeze(1) + 0.5 * log_variance
            elif FLAGS.mask_dim == 3:
                corr_loss = 15 * torch.exp(-0.5 * log_variance) * diff + 0.5 * log_variance
            else:
                raise NotImplementedError
            corr_loss = torch.sum(corr_loss * gt_mask, dim=[1,2,3]) / (torch.sum(gt_mask, dim=[1,2,3]) + 1e-5)
        elif FLAGS.use_uncertainty_loss and FLAGS.uncertain_type == 'ga':
            corr_loss = 0.5 * torch.exp(-log_variance) * diff**2 + 0.5 * log_variance
            corr_loss = torch.sum(corr_loss * gt_mask, dim=[1, 2, 3]) / (torch.sum(gt_mask, dim=[1, 2, 3]) + 1e-5)
        else:
            lower_corr_loss = torch.pow(diff, 2) / (2.0 * self.threshold)
            higher_corr_loss = diff - self.threshold / 2.0
            corr_loss_matrix = torch.where(diff > self.threshold, higher_corr_loss, lower_corr_loss)
            corr_loss_matrix = gt_mask * corr_loss_matrix
            corr_loss = torch.sum(corr_loss_matrix, dim=[1,2,3]) / (torch.sum(gt_mask, dim=[1,2,3]) + 1e-5)
        return corr_loss.mean()


    def cal_loss_Rot1(self, pred_v, gt_v):
        res = self.loss_func(pred_v, gt_v)
        return res.mean()

    def cal_loss_Rot2(self, pred_v, gt_v, sym):
        res = self.loss_func(pred_v, gt_v)
        valid_mask = sym[:, 0] == 0
        resw_valid = res[valid_mask]
        if resw_valid.shape[0] > 0:
            return resw_valid.mean()
        else:
            return torch.zeros(1, device=pred_v.device).squeeze()

    def cal_cosine_dis(self, pred_v, gt_v, sym=None):
        # pred_v  bs x 6, gt_v bs x 6
        res = (1.0 - torch.sum(pred_v * gt_v, dim=1)) * 2.0
        if sym is None:
            return torch.mean(res)
        else:
            valid_mask = sym[:, 0] == 0
            resw_valid = res[valid_mask]
            if resw_valid.shape[0] > 0:
                return resw_valid.mean()
            else:
                return torch.zeros(1, device=pred_v.device).squeeze()


    def cal_rot_regular_angle(self, pred_v1, pred_v2, sym):
        bs = pred_v1.shape[0]
        res = torch.zeros(1, device=pred_v1.device).squeeze()
        valid = 0.0
        for i in range(bs):
            if sym[i, 0] == 1:
                continue
            y_direction = pred_v1[i]
            z_direction = pred_v2[i]
            residual = torch.dot(y_direction, z_direction)
            res += torch.abs(residual)
            valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_loss_tran_scale(self, pred_v, gt_v):
        res = self.loss_func(pred_v, gt_v)
        return res.mean()

    def point_matching_loss(self, points, p_rot, p_t, g_rot, g_t):
        # Notice that this loss function do not back-propagate the grad of f_g_vec and f_r_vec
        # bs = points.shape[0]
        points = points.permute(0, 2, 1)
        pred_points = torch.bmm(p_rot, points) # + p_t[..., None]
        gt_points = torch.bmm(g_rot, points) # + g_t[..., None]
        return self.loss_func(pred_points, gt_points).mean()


def symmetry_rotation_matrix_y(number=30):
    result = []
    for i in range(number):
        theta = 2 * np.pi / number * i
        r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        result.append(r)
    result = np.stack(result)
    return result


def get_closest_rot(rot_est, rot_gt, sym_info):
    """get the closest rot_gt given rot_est and sym_info.

    rot_est: ndarray
    rot_gt: ndarray
    sym_info: None or Kx3x3 ndarray, m2m
    """
    if sym_info is None:
        return rot_gt
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    # find the closest rot_gt with smallest re
    r_err = re(rot_est, rot_gt)
    closest_rot_gt = rot_gt
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        rot_gt_sym = rot_gt.dot(sym_info[i])
        cur_re = re(rot_est, rot_gt_sym)
        if cur_re < r_err:
            r_err = cur_re
            closest_rot_gt = rot_gt_sym

    return closest_rot_gt

class Scale_loss(nn.Module):
    def __init__(self):
        super(Scale_loss, self).__init__()
        if FLAGS.pose_loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='none')
        elif FLAGS.pose_loss_type == 'smoothl1':   # same as MSE
            self.loss_func = nn.SmoothL1Loss(beta=0.5, reduction='none')
    def forward(self, pred_scale, gt_scale):
        # device = pred_scale.device
        # gt_scale = gt_scale.to(device)
        loss = self.loss_func(pred_scale, gt_scale)
        return loss.mean()

def get_closest_rot_batch(pred_rots, gt_rots, sym_infos):
    """
    get closest gt_rots according to current predicted poses_est and sym_infos
    --------------------
    pred_rots: [B, 4] or [B, 3, 3]
    gt_rots: [B, 4] or [B, 3, 3]
    sym_infos: list [Kx3x3 or None],
        stores K rotations regarding symmetries, if not symmetric, None
    -----
    closest_gt_rots: [B, 3, 3]
    """
    batch_size = pred_rots.shape[0]
    device = pred_rots.device

    closest_gt_rots = gt_rots.clone().cpu().numpy()  # B,3,3

    for i in range(batch_size):
        if sym_infos[i] is None:
            closest_gt_rots[i] = gt_rots[i].cpu().numpy()
        else:
            closest_rot = get_closest_rot(
                pred_rots[i].detach().cpu().numpy(),
                gt_rots[i].cpu().numpy(),
                sym_infos[i],
            )
            closest_gt_rots[i] = closest_rot
    closest_gt_rots = torch.tensor(closest_gt_rots, device=device, dtype=gt_rots.dtype)
    return closest_gt_rots

def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    return rd_deg


def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, mask, balance_weight=10, reduction='mean', sum_last_dim=True):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    if sum_last_dim:
        loss = balance_weight * 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target).sum(1).unsqueeze(1) + 0.5 * log_variance
    else:
        loss = balance_weight * 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5 * log_variance
    loss = loss * mask if mask is not None else loss
    return loss.mean() if reduction == 'mean' else loss.sum()


def gaussian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Neuips'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()
