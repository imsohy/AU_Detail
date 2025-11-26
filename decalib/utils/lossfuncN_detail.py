import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from functools import reduce
import torchvision.models as models
import cv2
import torchfile
from torch.autograd import Variable

from . import util
#Debug
def _dbg_finite(name: str, t):
    """Print finiteness and basic stats without changing tensors."""
    try:
        import torch
        with torch.no_grad():
            if t is None:
                print(f"[NaNGuard] {name}: None")
                return
            if not torch.is_tensor(t):
                print(f"[NaNGuard] {name}: not a tensor ({type(t)})")
                return
            finite = torch.isfinite(t)
            n_all = t.numel()
            n_finite = finite.sum().item()
            n_nonfinite = n_all - n_finite
            has_nan = torch.isnan(t).any().item()
            has_posinf = torch.isposinf(t).any().item() if hasattr(torch, "isposinf") else int(((~finite) & (t>0)).any().item())
            has_neginf = torch.isneginf(t).any().item() if hasattr(torch, "isneginf") else int(((~finite) & (t<0)).any().item())
            msg = (f"[NaNGuard] {name}: shape={tuple(t.shape)}, dtype={t.dtype}, dev={t.device}, "
                   f"nonfinite={n_nonfinite}/{n_all}, nan={int(has_nan)}, +inf={int(has_posinf)}, -inf={int(has_neginf)}")
            print(msg)
            if n_finite > 0:
                tt = t[finite]
                # robust stats (avoid very large prints)
                tmin = tt.min().item()
                tmax = tt.max().item()
                tmean = tt.mean().item()
                print(f"[NaNGuard] {name}: min={tmin:.6e}, max={tmax:.6e}, mean={tmean:.6e}")
            else:
                print(f"[NaNGuard] {name}: all elements are non-finite")
    except Exception as e:
        print(f"[NaNGuard] {_dbg_finite.__name__} error on {name}: {e}")


def weighted_au_landmark_loss(predicted_landmarks, landmarks_gt, au, au_weight, weight=5.):
    real_2d = landmarks_gt.clone()
    real_2d[:, :, 2] = 1.0

    # print('weight...',real_2d.shape)
    weights = torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1],), device=real_2d.device)

    # AU - landmark part
    for i in range(landmarks_gt.shape[0]):
        if True in (0.5 <= au[i, au_weight.brow_au]):
            weights[i, au_weight.brow] = weight / len(au_weight.brow)

        if True in (0.5 <= au[i, au_weight.brow_inner_au]):
            weights[i, au_weight.brow_inner] = weight / len(au_weight.brow_inner)

        if True in (0.5 <= au[i, au_weight.brow_outer_au]):
            weights[i, au_weight.brow_outer] = weight / len(au_weight.brow_outer)

        if True in (0.5 <= au[i, au_weight.eye_up_au]):
            weights[i, au_weight.eye_up] = weight / len(au_weight.eye_up)

        if True in (0.5 <= au[i, au_weight.eye_low_au]):
            weights[i, au_weight.eye_low] = weight / len(au_weight.eye_low)

        if True in (0.5 <= au[i, au_weight.eye_all_au]):
            weights[i, au_weight.eye_all] = weight / len(au_weight.eye_all)

        if True in (0.5 <= au[i, au_weight.nose_au]):
            weights[i, au_weight.nose] = weight / len(au_weight.nose)

        if True in (0.5 <= au[i, au_weight.lip_up_au]):
            weights[i, au_weight.lip_up] = weight / len(au_weight.lip_up)

        if True in (0.5 <= au[i, au_weight.lip_end_au]):
            weights[i, au_weight.lip_end] = weight / len(au_weight.lip_end)

        if True in (0.5 <= au[i, au_weight.mouth_au]):
            weights[i, au_weight.mouth] = weight / len(au_weight.mouth)

    weights[:, au_weight.lip_out] /= 5.0

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight


def related_au_landmark_loss(predicted_landmarks, landmarks_gt, au_weights, weight=1.):
    # print('real_2d_2',real_2d.shape)
    # weights = torch.ones((68,),device=predicted_landmarks.device)
    real_2d = landmarks_gt[:, :, :2]
    gt_distances = au_weights.au_related_landmark_distance(real_2d)
    pred_distances = au_weights.au_related_landmark_distance(predicted_landmarks)
    loss = 0.
    for i in range(len(gt_distances)):
        loss += (gt_distances[i] - pred_distances[i]).abs().mean()
    return loss

    # lip_right = landmarks[:,[65, 72, 73,  85, 71, 70, 69, 87, 88, 89, 103,    86, 74, 75, 76, 66, 94, 93, 92, 104], :]
    # lip_left = landmarks[:,[61, 90, 91,    82, 80, 84, 77, 95, 102, 98, 100,   81, 79, 83, 78, 67, 96, 101, 97, 99],:]

    # dis = torch.sqrt(((lip_right - lip_left)**2).sum(2)) #[bz, 4]


# [276 282 283 285 293 295 296 300 334 336  46  52  53  55  63  65  66  70
# 105 107 249 263 362 373 374 380 381 382 384 385 386 387 388 390 398 466
#   7  33 133 144 145 153 154 155 157 158 159 160 161 163 173 246 168   6
# 197 195   5   4 129  98  97   2 326 327 358   0  13  14  17  37  39  40
#  61  78  80  81  82  84  87  88  91  95 146 178 181 185 191 267 269 270
# 291 308 310 311 312 314 317 318 321 324 375 402 405 409 415]

def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()


### VAE
def kl_loss(texcode):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mu, logvar = texcode[:, :128], texcode[:, 128:]
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return KLD


### ------------------------------------- Losses/Regularizations for shading
# white shading
# uv_mask_tf = tf.expand_dims(tf.expand_dims(tf.constant( self.uv_mask, dtype = tf.float32 ), 0), -1)
# mean_shade = tf.reduce_mean( tf.multiply(shade_300W, uv_mask_tf) , axis=[0,1,2]) * 16384 / 10379
# G_loss_white_shading = 10*norm_loss(mean_shade,  0.99*tf.ones([1, 3], dtype=tf.float32), loss_type = "l2")
def shading_white_loss(shading):
    '''
    regularize lighting: assume lights close to white
    '''
    # rgb_diff = (shading[:,0] - shading[:,1])**2 + (shading[:,0] - shading[:,2])**2 + (shading[:,1] - shading[:,2])**2
    # rgb_diff = (shading[:,0].mean([1,2]) - shading[:,1].mean([1,2]))**2 + (shading[:,0].mean([1,2]) - shading[:,2].mean([1,2]))**2 + (shading[:,1].mean([1,2]) - shading[:,2].mean([1,2]))**2
    # rgb_diff = (shading.mean([2, 3]) - torch.ones((shading.shape[0], 3)).float().cuda())**2
    rgb_diff = (shading.mean([0, 2, 3]) - 0.99) ** 2
    return rgb_diff.mean()


def shading_smooth_loss(shading):
    '''
    assume: shading should be smooth
    ref: Lifting AutoEncoders: Unsupervised Learning of a Fully-Disentangled 3D Morphable Model using Deep Non-Rigid Structure from Motion
    '''
    dx = shading[:, :, 1:-1, 1:] - shading[:, :, 1:-1, :-1]
    dy = shading[:, :, 1:, 1:-1] - shading[:, :, :-1, 1:-1]
    gradient_image = (dx ** 2).mean() + (dy ** 2).mean()
    return gradient_image.mean()


### ------------------------------------- Losses/Regularizations for albedo
# texture_300W_labels_chromaticity = (texture_300W_labels + 1.0)/2.0
# texture_300W_labels_chromaticity = tf.divide(texture_300W_labels_chromaticity, tf.reduce_sum(texture_300W_labels_chromaticity, axis=[-1], keep_dims=True) + 1e-6)


# w_u = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :-1, :, :] - texture_300W_labels_chromaticity[:, 1:, :, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :-1, :, :] )
# G_loss_local_albedo_const_u = tf.reduce_mean(norm_loss( albedo_300W[:, :-1, :, :], albedo_300W[:, 1:, :, :], loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_u) / tf.reduce_sum(w_u+1e-6)


# w_v = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :, :-1, :] - texture_300W_labels_chromaticity[:, :, 1:, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :, :-1, :] )
# G_loss_local_albedo_const_v = tf.reduce_mean(norm_loss( albedo_300W[:, :, :-1, :], albedo_300W[:, :, 1:, :],  loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_v) / tf.reduce_sum(w_v+1e-6)

# G_loss_local_albedo_const = (G_loss_local_albedo_const_u + G_loss_local_albedo_const_v)*10

def albedo_constancy_loss(albedo, alpha=15, weight=1.):
    '''
    for similarity of neighbors
    ref: Self-supervised Multi-level Face Model Learning for Monocular Reconstruction at over 250 Hz
        Towards High-fidelity Nonlinear 3D Face Morphable Model
    '''
    albedo_chromaticity = albedo / (torch.sum(albedo, dim=1, keepdim=True) + 1e-6)
    weight_x = torch.exp(-alpha * (albedo_chromaticity[:, :, 1:, :] - albedo_chromaticity[:, :, :-1, :]) ** 2).detach()
    weight_y = torch.exp(-alpha * (albedo_chromaticity[:, :, :, 1:] - albedo_chromaticity[:, :, :, :-1]) ** 2).detach()
    albedo_const_loss_x = ((albedo[:, :, 1:, :] - albedo[:, :, :-1, :]) ** 2) * weight_x
    albedo_const_loss_y = ((albedo[:, :, :, 1:] - albedo[:, :, :, :-1]) ** 2) * weight_y

    albedo_constancy_loss = albedo_const_loss_x.mean() + albedo_const_loss_y.mean()
    return albedo_constancy_loss * weight


def albedo_ring_loss(texcode, ring_elements, margin, weight=1.):
    """
        computes ring loss for ring_outputs before FLAME decoder
        Inputs:
          ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
          Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
          Aim is to force each row (same subject) of each stream to produce same shape
          Each row of first N-1 strams are of the same subject and
          the Nth stream is the different subject
    """
    tot_ring_loss = (texcode[0] - texcode[0]).sum()
    diff_stream = texcode[-1]
    count = 0.0
    for i in range(ring_elements - 1):
        for j in range(ring_elements - 1):
            pd = (texcode[i] - texcode[j]).pow(2).sum(1)
            nd = (texcode[i] - diff_stream).pow(2).sum(1)
            tot_ring_loss = torch.add(tot_ring_loss,
                                      (torch.nn.functional.relu(margin + pd - nd).mean()))
            count += 1.0

    tot_ring_loss = (1.0 / count) * tot_ring_loss
    return tot_ring_loss * weight


def albedo_same_loss(albedo, ring_elements, weight=1.):
    """
        computes ring loss for ring_outputs before FLAME decoder
        Inputs:
          ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
          Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
          Aim is to force each row (same subject) of each stream to produce same shape
          Each row of first N-1 strams are of the same subject and
          the Nth stream is the different subject
    """
    loss = 0
    for i in range(ring_elements - 1):
        for j in range(ring_elements - 1):
            pd = (albedo[i] - albedo[j]).pow(2).mean()
            loss += pd
    loss = loss / ring_elements
    return loss * weight


### ------------------------------------- Losses/Regularizations for vertices
def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        real_2d_kp[:, :, 2] = weights[None, :] * real_2d_kp[:, :, 2]

    kp_gt = real_2d_kp.view(-1, 3)
    # print('true_content1_1_1', real_2d_kp[0][1][1])
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    # print('kpt_content1_1_1', kp_pred[0][1][1])
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8

    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k


def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # print('real_2d_1',landmarks_gt.shape)
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1], 1)).cuda()], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()

    # print('real_2d_2',real_2d.shape)
    real_2d = landmarks_gt
    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight


def landmark_HRNet_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # print('real_2d_1',landmarks_gt.shape)
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1], 1)).cuda()], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()

    # print('real_2d_2',real_2d.shape)
    weights = torch.zeros((68,)).to(landmarks_gt.device)
    weights[:17] = 1.5
    weights[5:7] = 3
    weights[10:12] = 3
    real_2d = landmarks_gt
    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight


# def eye_dis(landmarks):
#     # left eye:  [38,42], [39,41] - 1
#     # right eye: [44,48], [45,47] -1
#     eye_up = landmarks[:,[37, 38, 43, 44], :]
#     eye_bottom = landmarks[:,[41, 40, 47, 46], :]
#     dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]
#     return dis

# new eye loss
def eye_dis(landmarks):
    # # left eye:  [38,42], [39,41] - 1
    # # right eye: [44,48], [45,47] -1
    # eye_up = landmarks[:,[37, 38, 43, 44], :]
    # eye_bottom = landmarks[:,[41, 40, 47, 46], :]
    # tx = [21, 22, 23, 31, 25, 29, 37, 38, 41, 45, 39, 47]
    # dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]
    #
    # f = 2*torch.sqrt(((landmarks[:,[36,36,42,42]] - landmarks[:,[39,39,45,45], :])**2).sum(2))
    # # print(dis, f)
    # tx = [21, 22, 23, 31, 25, 29, 37, 38, 41, 45, 39, 47]
    eye_up = landmarks[:, [23, 25, 41, 39], :]
    eye_bottom = landmarks[:, [31, 29, 45, 47], :]
    dis = torch.sqrt(((eye_up - eye_bottom) ** 2).sum(2))  # [bz, 4]

    f = 2 * torch.sqrt(((landmarks[:, [21, 21, 38, 38]] - landmarks[:, [22, 22, 37, 37], :]) ** 2).sum(2))
    # print(dis, f)
    return dis / f


def eyed_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1], 1)).cuda()], dim=-1)
    real_2d = landmarks_gt
    pred_eyed = eye_dis(predicted_landmarks[:, :, :2])
    gt_eyed = eye_dis(real_2d[:, :, :2])

    loss = (pred_eyed - gt_eyed).abs().sum() / 2

    return loss


#
#
# [276 282 283 285 293 295 296 300 334 336  46  52  53  55  63  65  66  70
# 105 107 249 263 362 373 374 380 381 382 384 385 386 387 388 390 398 466
#   7  33 133 144 145 153 154 155 157 158 159 160 161 163 173 246 168   6
# 197 195   5   4 129  98  97   2 326 327 358   0  13  14  17  37  39  40
#  61  78  80  81  82  84  87  88  91  95 146 178 181 185 191 267 269 270
# 291 308 310 311 312 314 317 318 321 324 375 402 405 409 415]
def lip_dis(landmarks):
    # up inner lip:  [62, 63, 64] - 1
    # down innder lip: [68, 67, 66] -1
    # lip_up = landmarks[:,[61, 62, 63], :]
    # lip_down = landmarks[:,[67, 66, 65], :]
    # kk = [61, 291, 78, 308, 185, 40, 39, 37, 0, 267, 269, 270, 409,      191, 80, 81, 82, 13, 312, 311, 310, 415,    95, 88, 178, 87, 14, 317, 402, 318, 324,   146, 91, 181, 84, 17, 314, 405, 321, 375]
    # t = [72, 90, 73, 91,     85, 71, 70, 69, 65, 87, 88, 89, 103,        86, 74, 75, 76, 66, 94, 93, 92, 104,     81, 79, 83, 78, 67, 96, 101, 97, 99,           82, 80, 84, 77, 68, 95, 102, 98, 100]

    lip_right = landmarks[:, [65, 72, 73, 85, 71, 70, 69, 87, 88, 89, 103, 86, 74, 75, 76, 66, 94, 93, 92, 104], :]
    lip_left = landmarks[:, [61, 90, 91, 82, 80, 84, 77, 95, 102, 98, 100, 81, 79, 83, 78, 67, 96, 101, 97, 99], :]
    # lip_up = landmarks[:,[49, 50, 51,52,53, 61,62,63], :]
    # lip_down = landmarks[:,[59, 58, 57, 56, 55, 67, 66, 65], :]
    # dis = torch.sqrt(((lip_up - lip_down)**2).sum(2)) #[bz, 4]
    dis = torch.sqrt(((lip_right - lip_left) ** 2).sum(2))  # [bz, 4]
    return dis


def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1], 1)).cuda()], dim=-1)
    real_2d = landmarks_gt
    pred_lipd = lip_dis(predicted_landmarks[:, :, :2])
    gt_lipd = lip_dis(real_2d[:, :, :2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss


# new one for mouth
# def rel_dis(landmarks):
#     # lip_right = landmarks[:,[57, 51, 48, 60, 61, 62, 63], :]
#     # lip_left = landmarks[:,[8, 33, 54, 64, 67, 66, 65],:]
#     lip_right = landmarks[:,[57, 51, 48, 60, 61, 62, 63, 49, 50, 51, 52, 53], :]
#     lip_left = landmarks[:,[8, 33, 54, 64, 67, 66, 65, 59, 58, 57, 56, 55],:]
#
#     dis = torch.sqrt(((lip_right - lip_left)**2).sum(2)) #[bz, 4]
#     return dis
#
def rel_dis(landmarks):

    lip_right = landmarks[:, [57, 51, 48, 60, 61, 62, 63], :]
    lip_left = landmarks[:, [8, 33, 54, 64, 67, 66, 65], :]

    # lip_right = landmarks[:, [61, 62, 63], :]
    # lip_left = landmarks[:, [67, 66, 65], :]

    dis = torch.sqrt(((lip_right - lip_left) ** 2).sum(2))  # [bz, 4]

    return dis

def relative_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt)#.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=predicted_landmarks.device) #.cuda()
                             ], dim=-1)
    pred_lipd = rel_dis(predicted_landmarks[:, :, :2])
    gt_lipd = rel_dis(real_2d[:, :, :2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    # loss = F.mse_loss(pred_lipd, gt_lipd)

    return loss.mean()
# def relative_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
#     # if torch.is_tensor(landmarks_gt) is not True:
#     #     real_2d = torch.cat(landmarks_gt)#.cuda()
#     # else:
#     #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0],  landmarks_gt.shape[1], 1)).to(device=predicted_landmarks.device) #.cuda()
#     #                          ], dim=-1)
#     real_2d = landmarks_gt
#     pred_lipd = rel_dis(predicted_landmarks[:, :, :2])
#     gt_lipd = rel_dis(real_2d[:, :, :2])
#
#     loss = (pred_lipd - gt_lipd).abs().mean()
#     # loss = F.mse_loss(pred_lipd, gt_lipd)
#
#     return loss.mean()

# new one
def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # smaller inner landmark weights
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # import ipdb; ipdb.set_trace()
    real_2d = landmarks_gt

    # print('weight...',real_2d.shape)
    weights = torch.ones((landmarks_gt.shape[1],)).to(landmarks_gt.device)*1.5
    # weights[:20] = 3
    # weights[72:105] = 3
    # nose points
    #weights[52:65] = 1.5
    # weights[57] = 3  # version 6 is 3
    # weights[58] = 3
    # weights[61] = 3
    # weights[64] = 3
    # inner mouth
    # mouth_coner

    for m in [21,22,37, 38, 65,66,67, 68, 72,73,90,91]:
        weights[m] = 3
    weights[46]=2
    weights[40]=2
    weights[30]=2
    weights[24]=2

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight


# def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
#     #smaller inner landmark weights
#     # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
#     # import ipdb; ipdb.set_trace()
#     real_2d = landmarks_gt
#     weights = torch.ones((68,)).cuda()
#     weights[5:7] = 2
#     weights[10:12] = 2
#     # nose points
#     weights[27:36] = 1.5
#     weights[30] = 3
#     weights[31] = 3
#     weights[35] = 3
#     # inner mouth
#     weights[60:68] = 1.5
#     weights[48:60] = 1.5
#     weights[48] = 3
#     weights[54] = 3
#
#     loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
#     return loss_lmk_2d * weight

def landmark_loss_tensor(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight

from ..models.frnet import resnet50, load_state_dict
class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_model, pretrained_data='vggface2', device='cuda:1'):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval().to(device)
        load_state_dict(self.reg_model, pretrained_model)
        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).to(device)

    def reg_features(self, x):
        # out = []
        margin=10
        x = x[:,:,margin:224-margin,margin:224-margin]
        # x = F.interpolate(x*2. - 1., [224,224], mode='nearest')
        x = F.interpolate(x*2. - 1., [224,224], mode='bilinear')
        # import ipdb; ipdb.set_trace()
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # import ipdb;ipdb.set_trace()
        img = img[:, [2,1,0], :, :].permute(0,2,3,1) * 255 - self.mean_bgr
        img = img.permute(0,3,1,2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        # loss = ((gen_out - tar_out)**2).mean()
        loss = self._cos_metric(gen_out, tar_out).mean()
        return loss

class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x/self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        # print([x for x in out])
        return out

class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum.clamp_min(1e-6)

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        # dtype별 안전한 epsilon 사용 (fp32면 ~1e-7, fp16이면 더 큼)
        eps = max(1e-8, torch.finfo(cdist.dtype).eps * 10)

        #_dbg_finite("mrf/compute_relative_distances/cdist_in", cdist)

        # 1) 분자: 음수만 0으로 보정 (부동소수 오차 -1e-4 같은 값 제거)
        cdist = cdist.clamp_min(0.0)

        # 2) 분모: 채널축(=dim=1) 최소값을 ε 이상으로 보정 (단 한 번만 ε 적용)
        div = torch.amin(cdist, dim=1, keepdim=True).clamp_min(eps)
        #_dbg_finite("mrf/compute_relative_distances/div", div)

        relative_dist = cdist / div
        #_dbg_finite("mrf/compute_relative_distances/relative_dist_out", relative_dist)

        # (선택) 혹시 모를 폭주 방지
        # relative_dist = torch.nan_to_num(relative_dist, nan=0.0, posinf=1e6, neginf=0.0)

        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        #_dbg_finite("mrf/exp_norm_relative_dist/relative_dist_in", relative_dist)
        scaled = (self.bias - relative_dist) / self.nn_stretch_sigma
        # exp overflow 방지: 상한 clip (±80도 충분하지만 50이면 더 보수적)
        scaled = torch.clamp(scaled, max=50.0)
        dist_before_norm = torch.exp(scaled)
        #_dbg_finite("mrf/exp_norm_relative_dist/dist_before_norm", dist_before_norm)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        #_dbg_finite("mrf/exp_norm_relative_dist/cs_NCHW_out", self.cs_NCHW)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        # Debug
        #_dbg_finite("mrf/gen_feat_in", gen)
        #_dbg_finite("mrf/tar_feat_in", tar)
        """_dbg
        try:
            print(f"[NaNGuard] mrf/feat_shapes: gen={tuple(gen.shape)}, tar={tuple(tar.shape)}")
        except:
            pass
        """
        meanT = torch.mean(tar, 1, keepdim=True)
        # Debug
        #_dbg_finite("mrf/meanT", meanT)

        gen_feats, tar_feats = gen - meanT, tar - meanT
        # Debug
        #_dbg_finite("mrf/gen_feats", gen_feats)
        #_dbg_finite("mrf/tar_feats", tar_feats)

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)
        # Debug
        #_dbg_finite("mrf/gen_feats_norm", gen_feats_norm)
        #_dbg_finite("mrf/tar_feats_norm", tar_feats_norm)

        gen_normalized = gen_feats / gen_feats_norm.clamp_min(1e-6)
        tar_normalized = tar_feats / tar_feats_norm.clamp_min(1e-6)

        # Debug
        #_dbg_finite("mrf/gen_normalized", gen_normalized)
        #_dbg_finite("mrf/tar_normalized", tar_normalized)

        # Debug (원래 찍던 이름 유지 + 보완)
        #_dbg_finite("mrf/gen_feat_after_norm?", gen_feats_norm)
        #_dbg_finite("mrf/tar_feat_after_norm?", tar_feats_norm)

        cosine_dist_l = []
        BatchSize = tar.size(0)
        epsilon = 1e-5

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            # Debug
            #_dbg_finite(f"mrf/tar_feat_i[{i}]", tar_feat_i)
            #_dbg_finite(f"mrf/gen_feat_i[{i}]", gen_feat_i)

            patches_OIHW = self.patch_extraction(tar_feat_i)
            # Debug
            #_dbg_finite(f"mrf/patches_OIHW[{i}]", patches_OIHW)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            # Debug
            #_dbg_finite(f"mrf/cosine_dist_i[{i}]", cosine_dist_i)

            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist = torch.clamp(cosine_dist, -1.0, 1.0)

        # Debug
        #_dbg_finite("mrf/cosine_dist_cat", cosine_dist)

        cosine_dist_zero_2_one = (1.0 - cosine_dist) * 0.5
        cosine_dist_zero_2_one = torch.clamp(cosine_dist_zero_2_one, 0.0, 1.0)

        # Debug
        #_dbg_finite("mrf/cosine_zero2one", cosine_dist_zero_2_one)

        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        # Debug
        #_dbg_finite("mrf/relative_dist", relative_dist)

        rela_dist = self.exp_norm_relative_dist(relative_dist)
        # Debug
        #_dbg_finite("mrf/rela_dist", rela_dist)

        dims_div_mrf = rela_dist.size()

        """_dbg
        try:
            print(f"[NaNGuard] mrf/rela_dist_shape={tuple(dims_div_mrf)}")
        except:
            pass
        """

        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        # Debug
        #_dbg_finite("mrf/k_max_nc", k_max_nc)

        div_mrf = torch.mean(k_max_nc, dim=1)
        # Debug
        #_dbg_finite("mrf/div_mrf_preclamp", div_mrf)

        div_mrf = div_mrf.clamp_min(1e-6)
        # Debug
        #_dbg_finite("mrf/div_mrf_postclamp", div_mrf)

        div_mrf_sum = -torch.log(div_mrf)
        # Debug
        #_dbg_finite("mrf/div_mrf_sum_postlog", div_mrf_sum)

        div_mrf_sum = torch.sum(div_mrf_sum)
        # Debug
        #_dbg_finite("mrf/div_mrf_sum_reduced", div_mrf_sum)

        return div_mrf_sum

    def forward(self, gen, tar):
        # Debug
        #_dbg_finite("mrf/gen_in", gen)
        #_dbg_finite("mrf/tar_in", tar)

        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)

        # Debug
        """
        try:
            for k in gen_vgg_feats.keys():
                _dbg_finite(f"mrf/gen_feat[{k}]", gen_vgg_feats[k])
            for k in tar_vgg_feats.keys():
                _dbg_finite(f"mrf/tar_feat[{k}]", tar_vgg_feats[k])
        except Exception as _e:
            print(f"[NaNGuard] forward-feat dict iter error: {_e}")
        """
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style
        # Debug
        #_dbg_finite("mrf/style_loss_scalar", self.style_loss)

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

        # Debug
        #_dbg_finite("mrf/content_loss_scalar", self.content_loss)

        return self.style_loss + self.content_loss

        # loss = 0
        # for key in self.feat_style_layers.keys():
        #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
        # return loss