import torch
def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 2
    kp_pred: N x K x 2
    """
    kp_gt = real_2d_kp.view(-1,3)
    # print('true_content1_1_1', real_2d_kp[0][1][1])
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    # print('kpt_content1_1_1', kp_pred[0][1][1])
    vis = kp_gt[:,2]
    k = torch.sum(vis) * 2.0 + 1e-8

    dif_abs = torch.abs(kp_gt[:,:2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k

def transform_points(points, tform, points_scale=None, out_scale=None):
    points_2d = points[:, :, :2]

    # 'input points must use original range'
    if points_scale:
        assert points_scale[0] == points_scale[1]
        points_2d = (points_2d * 0.5 + 0.5) * points_scale[0]
    # import ipdb; ipdb.set_trace()

    batch_size, n_points, _ = points.shape
    tx = torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device='cpu', dtype=points.dtype)], dim=-1)

    trans_points_2d = torch.bmm(tx,tform)
    if out_scale:  # h,w of output image size
        trans_points_2d[:, :, 0] = trans_points_2d[:, :, 0] / out_scale[1] * 2 - 1
        trans_points_2d[:, :, 1] = trans_points_2d[:, :, 1] / out_scale[0] * 2 - 1
    trans_points = torch.cat([trans_points_2d[:, :, :2], points[:, :, 2:]], dim=-1)
    return trans_points