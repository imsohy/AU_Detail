import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from .utils import util
from .utils.config_wt_HWcode import cfg
from .utils import lossfuncN_detail as lossfunc
from .utils.util import au_weights
from .models.expression_loss import ExpressionLossNet
from .models.OpenGraphAU.model.MEFL import MEFARG
from .models.OpenGraphAU.utils import load_state_dict
from .models.OpenGraphAU.utils import *
from .models.OpenGraphAU.conf import get_config, set_logger, set_outdir, set_env
from .datasets import build_datasets_video_WT_detail as build_datasets

# from .datasets import build_datasets_image as build_datasets
# from .datasets import build_datasets_NoAug
torch.backends.cudnn.benchmark = True


def _finite(name, t):
    import torch
    assert torch.isfinite(t).all(), f"[NaNGuard] {name} has non-finite values"


class Trainer(object):
    def __init__(self, model, config=None, device='cuda:1'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.K = self.cfg.dataset.K
        # training stage: coarse and detail
        self.train_detail = self.cfg.train.train_detail
        self.vis_au = self.cfg.train.vis_au

        # mymodel model
        self.mymodel = model.to(self.device)

        # initialize detail loss object always
        self.mrf_loss = lossfunc.IDMRFLoss()
        self.face_attr_mask = util.load_local_mask(image_size=self.cfg.model.uv_size, mode='bbx')

        self.configure_optimizers()
        self.load_checkpoint()
        self.au_weight = au_weights()
        # if self.cfg.loss.weightedAU:
        # self.weightedAU = np.load('/mnt/hdd/EncoderTrainingCode/Code/data/AU_weight.npy')
        # self.weightedAU = torch.tensor(self.weightedAU).to(self.device)
        # self.au_weight = au_weights()

        # initialize loss
        # if self.train_detail:     
        #     self.mrf_loss = lossfunc.IDMRFLoss()
        #     self.face_attr_mask = util.load_local_mask(image_size=self.cfg.model.uv_size, mode='bbx')
        # else:
        #     self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)      

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(
            list(self.mymodel.BiViT.parameters()) +  # coarse+detail ViT
            list(self.mymodel.D_detail.parameters()),  # detail decoder
            lr=self.cfg.train.lr,
            amsgrad=False
        )

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.train.stepLR_steps, gamma=0.999)

    def load_checkpoint(self):
        # au config
        self.auconf = get_config()
        self.auconf.evaluate = True
        self.auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        set_env(self.auconf)
        model_dict = self.mymodel.model_dict()

        self.expression_net = ExpressionLossNet().to(self.device)
        emotion_checkpoint = torch.load(self.cfg.emotion_checkpoint)['state_dict']
        emotion_checkpoint['linear.0.weight'] = emotion_checkpoint['linear.weight']
        emotion_checkpoint['linear.0.bias'] = emotion_checkpoint['linear.bias']
        self.expression_net.load_state_dict(emotion_checkpoint, strict=False)
        self.expression_net.eval()
        self.AU_net = MEFARG(num_main_classes=self.auconf.num_main_classes, num_sub_classes=self.auconf.num_sub_classes,
                             backbone=self.auconf.arc).to(self.device)
        self.AU_net = load_state_dict(self.AU_net, self.auconf.resume).to(self.device)
        self.AU_net.eval()
        # resume training, including model weight, opt, steps
        # import ipdb; ipdb.set_trace()
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            # print('True')
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            model_dict = self.mymodel.model_dict()
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
                else:
                    continue
                    # print("check model path", os.path.join(self.cfg.output_dir, 'model.tar'))
                    # exit()
            util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            # self.opt.param_groups[0]['lr'] = 0.000005
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0
        # load model weights only
        if os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])

        if os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            key = 'E_flame'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            # self.global_step = 0
        else:
            logger.info('model path not found')

    def training_step(self, batch, batch_nb, training_type='coarse'):
        self.mymodel.eval()
        self.mymodel.BiViT.train()

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images_224 = batch['image_224'].to(self.device);
        images_224 = images_224.view(-1, images_224.shape[-3], images_224.shape[-2], images_224.shape[-1])
        # images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        lmk = batch['landmark'].to(self.device);
        lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        lmk_dense = batch['landmark_dense'].to(self.device);
        lmk_dense = lmk_dense.view(-1, lmk_dense.shape[-2], lmk_dense.shape[-1])
        # image_au = batch['au']
        # if len(image_au) != self.batch_size:
        # image_au = image_au.to(self.device)
        masks = batch['mask'].to(self.device);
        masks = masks.view(-1, images_224.shape[-3], images_224.shape[-2], images_224.shape[-1])
        # masks = batch['mask'].to(self.device); masks = masks.view(-1,images.shape[-3], images.shape[-2], images.shape[-1])
        # -- encoder
        codedict_old, codedict = self.mymodel.encode(images_224)
        # images = images_224
        images = images_224[1:2]
        lmk = lmk[1:2]
        lmk_dense = lmk_dense[1:2]
        masks = masks[1:2]
        batch_size = 1
        # batch_size = images_224.shape[0]

        ###--------------- training coarse & detail  model
        # -- decoder
        rendering = True
        opdict = self.mymodel.decode(codedict,
                                     codedict_old,
                                     rendering=rendering,
                                     vis_lmk=False,
                                     return_vis=False,
                                     use_detail=True)
        opdict['images'] = images
        opdict['lmk'] = lmk
        opdict['lmk_dense'] = lmk_dense

        if self.cfg.loss.photo > 0.:
            # ------ rendering
            # mask
            mask_face_eye = F.grid_sample(self.mymodel.uv_face_eye_mask.expand(batch_size, -1, -1, -1),
                                          opdict['grid'].detach(), align_corners=False)
            # images
            predicted_images = opdict['rendered_images'] * mask_face_eye * opdict['alpha_images']
            opdict['predicted_images'] = predicted_images

        #### ----------------------- Losses
        losses = {}

        ############################# base shape
        predicted_landmarks = opdict['landmarks2d']
        predicted_landmarks_dense = opdict['mp_landmark']
        if self.cfg.loss.AUFLoss > 0:
            _, afnT = self.mymodel.AUNet(images, use_gnn=False)
            _, afnP = self.mymodel.AUNet(opdict['rendered_images'], use_gnn=False)
            losses['AUFLoss'] = F.mse_loss(afnT.view(-1, 1), afnP.view(-1, 1)) * self.cfg.loss.expression

        # if self.cfg.loss.AUloss:
        #       image_au = self.AU_net(images)[1]
        #       rend_au = self.AU_net(opdict['rendered_images'])[1]
        if self.cfg.loss.weightedAU:
            # rend_au_loss = self.AU_net(opdict['rendered_images'])[1][0]*self.weightedAU
            image_au = self.AU_net(images)[1]
            rend_au = self.AU_net(opdict['rendered_images'])[1]
            losses['au_lmk_loss'] = lossfunc.weighted_au_landmark_loss(predicted_landmarks_dense, lmk_dense,
                                                                       image_au.float(),
                                                                       self.au_weight) * self.cfg.loss.mainAU
            losses['au_loss'] = F.mse_loss(image_au, rend_au)
            opdict['au_img'] = (image_au >= 0.5).float()
            opdict['au_rend'] = (rend_au >= 0.5).float()
        elif self.cfg.loss.focalAU:
            # if len(image_au) != self.batch_size:
            image_au = self.AU_net(images)
            rend_au = self.AU_net(opdict['rendered_images'])
            losses['au_lmk_loss'] = lossfunc.weighted_au_landmark_loss(predicted_landmarks_dense, lmk_dense,
                                                                       image_au[1].float(),
                                                                       self.au_weight) * self.cfg.loss.mainAU
            losses['au_dist_loss'] = lossfunc.related_au_landmark_loss(predicted_landmarks, lmk,
                                                                       self.au_weight) * self.cfg.loss.subAU
            losses['au_loss'] = F.mse_loss(image_au[1], rend_au[1])
            opdict['au_img'] = (image_au[1] >= 0.5).float()
            opdict['au_rend'] = (rend_au[1] >= 0.5).float()

        if self.cfg.loss.lmk_mp > 0:
            losses['landmark_dense_loss'] = lossfunc.weighted_landmark_loss(predicted_landmarks_dense,
                                                                            opdict['lmk_dense']) * self.cfg.loss.lmk_mp
        if self.cfg.loss.eyed > 0.:
            losses['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks_dense,
                                                        opdict['lmk_dense']) * self.cfg.loss.eyed
        if self.cfg.loss.lipd > 0.:
            losses['lip_distance'] = lossfunc.lipd_loss(predicted_landmarks_dense,
                                                        opdict['lmk_dense']) * self.cfg.loss.lipd

        if self.cfg.loss.lmk:
            losses['landmark'] = lossfunc.landmark_HRNet_loss(predicted_landmarks, opdict['lmk']) * self.cfg.loss.lmk
        if self.cfg.loss.relaL > 0.:
            losses['relative_landmark'] = lossfunc.relative_landmark_loss(predicted_landmarks,
                                                                          opdict['lmk']) * self.cfg.loss.relaL
        # losses['lmk_3d_reg'] = torch.mean(
        #     (opdict['landmarks3d_world'] - opdict['landmarks3d_world_deca']) ** 2) / 2 * self.cfg.loss.reg_3dlmk
        # losses['shape_reg'] = torch.mean((codedict['shape'] - codedict_deca['shape']) ** 2) / 2*self.cfg.loss.reg_shape_deca
        # losses['expression_reg'] = torch.mean((codedict['exp'] - codedict_deca['exp']) ** 2) / 2*self.cfg.loss.reg_exp_deca
        # losses['pose_reg'] = torch.mean((codedict['pose'] - codedict_deca['pose']) ** 2) / 2*self.cfg.loss.reg_pose_deca
        # losses['cam_reg'] = torch.mean((codedict['cam'] - codedict_deca['cam']) ** 2) / 2*self.cfg.loss.reg_cam_deca
        if self.cfg.loss.photo > 0.:
            if self.cfg.loss.useSeg:
                masks = masks[:, None, :, :]
            else:
                masks = mask_face_eye * opdict['alpha_images']
            losses['photometric_texture'] = (masks * (
                    predicted_images - opdict['images']).abs()).mean() * self.cfg.loss.photo

        if self.cfg.loss.expression > 0:
            faces_gt = images
            faces_pred = opdict['rendered_images']

            opdict['faces_gt'] = faces_gt
            opdict['faces_pred'] = faces_pred
            self.expression_net.eval()

            emotion_features_pred = self.expression_net(faces_pred)

            with torch.no_grad():
                emotion_features_gt = self.expression_net(faces_gt)

            losses['expression'] = F.mse_loss(emotion_features_pred, emotion_features_gt) * self.cfg.loss.expression

        # ============================ Detail Losses (DECA 방식) ============================
        # 1) GT UV 텍스처: 입력 이미지를 UV 좌표계로 샘플링
        uv_pverts = self.mymodel.render.world2uv(opdict['trans_verts'])  # (B, H, W, 3)
        uv_texture_gt = F.grid_sample(
            images,  # GT 이미지
            uv_pverts.permute(0, 2, 3, 1)[..., :2],  # UV 좌표
            mode='bilinear', align_corners=False
        )

        # 2) 예측 UV 텍스처 & visibility mask
        uv_texture_pred = opdict['uv_texture']  # decode(use_detail=True)에서 얻음
        uv_detail_normals = opdict['uv_detail_normals']
        uv_vis_mask = ((uv_detail_normals[:, [-1], :, :] < -0.05).float()
                       * self.mymodel.uv_face_eye_mask)  # -Z 노멀 & 얼굴 마스크

        # 3) 얼굴 패치 선택 & 리사이즈 (원본 DECA와 동일한 방식)
        pi = 0
        bb = self.face_attr_mask[pi]  # [x1, x2, y1, y2]
        new_size = 256
        uv_patch_pred = F.interpolate(
            uv_texture_pred[:, :, bb[2]:bb[3], bb[0]:bb[1]],
            [new_size, new_size], mode='bilinear'
        )
        uv_patch_gt = F.interpolate(
            uv_texture_gt[:, :, bb[2]:bb[3], bb[0]:bb[1]],
            [new_size, new_size], mode='bilinear'
        )
        uv_patch_mask = F.interpolate(
            uv_vis_mask[:, :, bb[2]:bb[3], bb[0]:bb[1]],
            [new_size, new_size], mode='bilinear'
        )
        #        _finite("uv_texture_pred", uv_texture_pred)
        #        _finite("uv_texture_gt", uv_texture_gt)
        #        _finite("uv_detail_normals", uv_detail_normals)
        #        _finite("uv_vis_mask", uv_vis_mask)
        #        _finite("uv_patch_pred", uv_patch_pred)
        #        _finite("uv_patch_gt", uv_patch_gt)
        #        _finite("uv_patch_mask", uv_patch_mask)

        # 4) Detail 손실 항목들 (DECA trainer.py와 같은 사용 패턴)
        # (a) photometric detail (L1)
        losses['photo_detail'] = (
                (uv_patch_pred * uv_patch_mask - uv_patch_gt * uv_patch_mask).abs().mean()
                * self.cfg.loss.photo_D
        )
        _m_in = uv_patch_pred * uv_patch_mask
        _t_in = uv_patch_gt * uv_patch_mask
        _finite("mrf_input_pred_masked", _m_in)
        _finite("mrf_input_gt_masked", _t_in)

        """
        # IDMRF safe version ... did not solve the problem.
        _vis_ratio = (uv_patch_mask > 0.5).float().mean()
        if (not torch.isfinite(_vis_ratio)) or _vis_ratio.item() < 0.02:
            losses['photo_detail_mrf'] = torch.zeros([], device=uv_patch_mask.device)
        else:
            losses['photo_detail_mrf'] = (
                    self.mrf_loss(uv_patch_pred * uv_patch_mask, uv_patch_gt * uv_patch_mask)
                    * self.cfg.loss.photo_D * self.cfg.loss.mrf
            )
        """
        # (b) ID-MRF detail
        losses['photo_detail_mrf'] = (
                self.mrf_loss(uv_patch_pred * uv_patch_mask, uv_patch_gt * uv_patch_mask)
                * self.cfg.loss.photo_D * self.cfg.loss.mrf
        )
        # (c) z_reg (displacement L1/L2 규제: 여기서는 L1)
        uv_z = opdict['displacement_map']  # (B, 1, H, W)
        losses['z_reg'] = uv_z.abs().mean() * self.cfg.loss.reg_z

        # (d) z_diff (shading smoothness)
        uv_shading = self.mymodel.render.add_SHlight(uv_detail_normals, codedict['light'])
        losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading) * self.cfg.loss.reg_diff

        # (e) z_sym (좌우 대칭 규제: 비가시 영역에 대해 강하게)
        if self.cfg.loss.reg_sym > 0.:
            nonvis_mask = (1 - util.binary_erosion(uv_vis_mask))
            losses['z_sym'] = (
                                      nonvis_mask * (uv_z - torch.flip(uv_z, dims=[-1]).detach()).abs()
                              ).sum() * self.cfg.loss.reg_sym
        # ================================================================================

        if self.cfg.loss.reg_shape > 0:
            losses['shape_reg_self'] = (torch.sum(codedict['shape'] ** 2) / 2) * self.cfg.loss.reg_shape
        losses['expression_reg_self'] = (torch.sum(codedict['exp'] ** 2) / 2) * self.cfg.loss.reg_exp
        losses['tex_reg'] = (torch.sum(codedict['tex'] ** 2) / 2) * self.cfg.loss.reg_tex
        losses['light_reg'] = ((torch.mean(codedict['light'], dim=2)[:, :, None] - codedict[
            'light']) ** 2).mean() * self.cfg.loss.reg_light
        # losses['tex_reg_self'] = (torch.sum(codedict['tex']**2)/2)*self.cfg.loss.reg_albedo
        # # print(codedict['pose'].shape)
        losses['jaw_pose_reg_self'] = (torch.sum(codedict['pose'][:, 3:] ** 2) / 2) * self.cfg.loss.reg_jaw_pose
        # losses['light_reg_self'] = ((torch.mean(codedict['light'], dim=2)[:,:,None] - codedict['light'])**2).mean()*self.cfg.loss.reg_light
        if self.cfg.model.jaw_type == 'euler':
            # import ipdb; ipdb.set_trace()
            # reg on jaw pose
            losses['reg_jawpose_roll'] = (torch.sum(codedict['euler_jaw_pose'][:, -1] ** 2) / 2) * 100.
            losses['reg_jawpose_close'] = (torch.sum(F.relu(-codedict['euler_jaw_pose'][:, 0]) ** 2) / 2) * 10.

        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict

    # def evaluate(self):
    #     ''' NOW validation
    #     '''
    #     os.makedirs(os.path.join(self.cfg.output_dir, 'NOW_validation'), exist_ok=True)
    #     savefolder = os.path.join(self.cfg.output_dir, 'NOW_validation', f'step_{self.global_step:08}')
    #     os.makedirs(savefolder, exist_ok=True)
    #     self.mymodel.eval()
    #     # run now validation images
    #     from .datasets.now import NoWDataset
    #     dataset = NoWDataset(scale=(self.cfg.dataset.scale_min + self.cfg.dataset.scale_max) / 2)
    #     dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
    #                             num_workers=8,
    #                             pin_memory=True,
    #                             drop_last=False)
    #     faces = self.mymodel.flame.faces_tensor.cpu().numpy()
    #     for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
    #         images = batch['image'].to(self.device)
    #         images_224 = batch['image_224'].to(self.device)
    #         imagename = batch['imagename']
    #         with torch.no_grad():
    #             codedict = self.mymodel.encode(images, images_224)
    #             _, visdict = self.mymodel.decode(codedict)
    #             codedict['exp'][:] = 0.
    #             codedict['pose'][:] = 0.
    #             opdict, _ = self.mymodel.decode(codedict)
    #         # -- save results for evaluation
    #         verts = opdict['verts'].cpu().numpy()
    #
    #         landmark_51 = opdict['landmarks3d_world'][:, 17:]
    #         landmark_7 = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]
    #         landmark_7 = landmark_7.cpu().numpy()
    #         for k in range(images.shape[0]):
    #             os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
    #             # save mesh
    #             util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
    #             # save 7 landmarks for alignment
    #             np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
    #             for vis_name in visdict.keys():  # ['inputs', 'landmarks2d', 'shape_images']:
    #                 if vis_name not in visdict.keys():
    #                     continue
    #                 # import ipdb; ipdb.set_trace()
    #                 image = util.tensor2image(visdict[vis_name][k])
    #                 name = imagename[k].split('/')[-1]
    #                 # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
    #                 cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name + '.jpg'), image)
    #         # visualize results to check
    #         util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))
    #
    #     ## then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
    #     self.mymodel.train()
    #     # self.mymodel.
    #     self.mymodel.E_flame_224.eval()
    # def validation_step(self):
    #     self.mymodel.eval()
    #     try:
    #         batch = next(self.val_iter)
    #     except:
    #         self.val_iter = iter(self.val_dataloader)
    #         batch = next(self.val_iter)
    #     images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    #     with torch.no_grad():
    #         codedict = self.mymodel.encode(images)
    #         opdict, visdict = self.mymodel.decode(codedict)
    #     savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.global_step:08}.jpg')
    #     grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
    #     self.writer.add_image('val_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)
    #     self.mymodel.train()

    # def evaluate(self):
    #     ''' NOW validation
    #     '''
    #     os.makedirs(os.path.join(self.cfg.output_dir, 'NOW_validation'), exist_ok=True)
    #     savefolder = os.path.join(self.cfg.output_dir, 'NOW_validation', f'step_{self.global_step:08}')
    #     os.makedirs(savefolder, exist_ok=True)
    #     self.mymodel.eval()
    #     # run now validation images
    #     from .datasets.now import NoWDataset
    #     dataset = NoWDataset(scale=(self.cfg.dataset.scale_min + self.cfg.dataset.scale_max)/2)
    #     dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
    #                         num_workers=8,
    #                         pin_memory=True,
    #                         drop_last=False)
    #     faces = self.mymodel.flame.faces_tensor.cpu().numpy()
    #     for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
    #         images = batch['image'].to(self.device)
    #         imagename = batch['imagename']
    #         with torch.no_grad():
    #             codedict = self.mymodel.encode(images)
    #             _, visdict = self.mymodel.decode(codedict)
    #             codedict['exp'][:] = 0.
    #             codedict['pose'][:] = 0.
    #             opdict, _ = self.mymodel.decode(codedict)
    #         #-- save results for evaluation
    #         verts = opdict['verts'].cpu().numpy()
    #         landmark_51 = opdict['landmarks3d_world'][:, 17:]
    #         landmark_7 = landmark_51[:,[19, 22, 25, 28, 16, 31, 37]]
    #         landmark_7 = landmark_7.cpu().numpy()
    #         for k in range(images.shape[0]):
    #             os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
    #             # save mesh
    #             util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
    #             # save 7 landmarks for alignment
    #             np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
    #             for vis_name in visdict.keys(): #['inputs', 'landmarks2d', 'shape_images']:
    #                 if vis_name not in visdict.keys():
    #                     continue
    #                 # import ipdb; ipdb.set_trace()
    #                 image = util.tensor2image(visdict[vis_name][k])
    #                 name = imagename[k].split('/')[-1]
    #                 # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
    #                 cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'), image)
    #         # visualize results to check
    #         util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))
    #
    #     ## then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
    #     self.mymodel.train()

    def prepare_data(self):
        # self.train_dataset = build_datasets_NoAug.build_train(self.cfg.dataset)
        self.train_dataset = build_datasets.build_train(self.cfg.dataset)
        logger.info('---- training data numbers: ', len(self.train_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.cfg.dataset.num_workers,
                                           pin_memory=True,
                                           drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        # self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=True,
        #                     num_workers=8,
        #                     pin_memory=True,
        #                     drop_last=False)
        # self.val_iter = iter(self.val_dataloader)

    def fit(self):
        self.prepare_data()

        iters_every_epoch = int(len(self.train_dataset) / self.batch_size)
        start_epoch = self.global_step // iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            # random.shuffle(self.train_dataset)
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch + 1}/{self.cfg.train.max_epochs}]"):
                if epoch * iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                losses, opdict = self.training_step(batch, step)
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/' + k, v, global_step=self.global_step)
                    logger.info(loss_info)
                    if self.global_step % self.cfg.train.vis_steps == 0 or step == (iters_every_epoch - 1):
                        # visind = list(range(self.cfg.dataset.batch_size* self.cfg.dataset.K)) #!!!!!!
                        visind = list(range(self.cfg.dataset.batch_size))
                        shape_images_full, shape_images_face = self.mymodel.render.render_shape(opdict['verts'][visind],
                                                                                                opdict['trans_verts'][
                                                                                                    visind],
                                                                                                images=opdict['images'][
                                                                                                    visind])
                        shape_images_full_old, shape_images_face_old = self.mymodel.render.render_shape(
                            opdict['verts_old'][visind], opdict['trans_verts_old'][visind],
                            images=opdict['images'][visind])
                        if self.vis_au:
                            visdict = {
                                'inputs': opdict['images'][visind],
                                # 'landmarks2d_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind], isScale=True),
                                # 'landmarks2d': util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind], isScale=True),
                                'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind],
                                                                            opdict['mp_landmark'][visind],
                                                                            opdict['lmk_dense'][visind], isScale=True),
                                # 'au_gt': util.draw_activation_circles(opdict['images'][visind], opdict['lmk_dense'][visind], opdict['au_img'], self.au_weight),
                                # 'au_pred': util.draw_activation_circles(opdict['rendered_images'][visind], opdict['mp_landmark'][visind], opdict['au_rend'], self.au_weight),
                                # 'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind], opdict['mp_landmark'][visind], isScale=True),
                                # 'shape_images': shape_images_face,
                                'shape_images_full': shape_images_full,
                                # 'au_rend': util.draw_activation_circles(opdict['rendered_images'], opdict['landmarks2d'][visind], opdict['au_rend']),
                                'rendered_images': opdict['rendered_images'],
                                'vis_au': util.vis_au(opdict['au_img'], opdict['au_rend'])
                                # 'predicted_images': opdict['predicted_images'][visind],
                            }
                        else:
                            visdict = {
                                'inputs': opdict['images'][visind],
                                # 'landmarks2d_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind], isScale=True),
                                # 'landmarks2d': util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind], isScale=True),
                                'shape_images_old': shape_images_face_old,
                                'shape_images_full_old': shape_images_full_old,
                                'landmarks_dens_gt': util.tensor_vis_landmarks(opdict['images'][visind],
                                                                               opdict['lmk_dense'][visind],
                                                                               isScale=True),
                                'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind],
                                                                            opdict['mp_landmark'][visind],
                                                                            isScale=True),
                                'shape_images': shape_images_face,
                                'shape_images_full': shape_images_full,
                                'rendered_images': opdict['rendered_images'],
                                # 'rendered_images': opdict['rendered_images']
                                # 'predicted_images': opdict['predicted_images'][visind],
                            }
                            if 'render_images_detail' in opdict:
                                visdict['render_images_detail'] = opdict['render_images_detail'][visind]

                        if 'shape_detail_images' in opdict:
                            visdict['shape_detail_images'] = opdict['shape_detail_images'][visind]
                        if 'shape_detail_images_full' in opdict:
                            visdict['shape_detail_images_full'] = opdict['shape_detail_images_full'][visind]
                        if 'predicted_detail_images' in opdict:
                            visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]
                        if 'render_images_detail' in opdict:
                            visdict['render_images_detail'] = opdict['render_images_detail'][visind]

                        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir,
                                                f'{self.global_step:06}.jpg')
                        grid_image = util.visualize_grid(visdict, savepath, return_gird=True)

                        # ----5장 가로 합성본: [global_step]_sum.jpg 저장 ----
                        try:
                            import torch.nn.functional as F
                            with torch.no_grad():
                                idx = visind[0] if isinstance(visind, list) and len(visind) > 0 else 0

                                # 1) 패널 소스 준비 (decode에서 넘겨준 것을 사용)
                                img_orig = opdict['images'][idx:idx + 1]  # 1) 원본

                                # 2) 세로 로직과 동일하게 render_shape 직접 호출
                                # face_colors(회색) * shading + original background
                                # 텍스처 albedo는 포함되지 않음 (얼굴 영역만 렌더링)
                                shape_images_full_for_sum, shape_images_face_for_sum = self.mymodel.render.render_shape(
                                    opdict['verts'][idx:idx + 1],
                                    opdict['trans_verts'][idx:idx + 1],
                                    images=opdict['images'][idx:idx + 1]
                                )
                                img_mesh = shape_images_face_for_sum  # 2) 코어스 메쉬(face_colors*shading + original bg)

                                # 3) coarse mesh + albedo texture + shading, 검은 배경
                                # rendered_images는 decode에서 render(background=None)로 생성됨
                                img_alb = opdict['rendered_images'][idx:idx + 1]  # 3) 알베도 텍스처 렌더링 (검은 배경)

                                # 4) detail mesh (회색) + 원본 배경
                                img_detail_mesh = opdict.get('shape_detail_images', img_mesh)[
                                                  idx:idx + 1]  # 4) 디테일+코어스 메쉬(얼굴 영역)

                                # 5) coarse+detail mesh + albedo texture (3번과 유사하지만 detail 포함)
                                # predicted_detail_images = albedo * detail_shading (검은 배경)
                                img_detail_albedo = opdict.get('predicted_detail_images',
                                                               opdict.get('rendered_images', img_alb))[idx:idx + 1]

                                # 텐서→u8
                                def to_u8(t):
                                    t = t.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                                    t = (t * 255.0).astype(np.uint8)
                                    # RGB -> BGR 변환 (cv2.imwrite는 BGR 순서를 기대함)
                                    return t[:, :, [2, 1, 0]]

                                im0 = to_u8(img_orig)  # 1) 원본
                                im1 = to_u8(img_mesh)  # 2) 코어스 메쉬(회색 + 원본 배경)
                                im2 = to_u8(img_alb)  # 3) 코어스 메쉬 + 알베도 렌더링(검은 배경)
                                im3 = to_u8(img_detail_mesh)  # 4) 디테일 메쉬(회색 + 원본 배경)
                                im4 = to_u8(img_detail_albedo)  # 5) 디테일 메쉬 + 알베도(검은 배경)

                                # 가로 합치기 & 저장 (높이 맞춤)
                                H = im0.shape[0]

                                def resize_h(im, H):
                                    if im.shape[0] == H: return im
                                    return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H),
                                                      interpolation=cv2.INTER_LINEAR)

                                im1 = resize_h(im1, H)
                                im2 = resize_h(im2, H)
                                im3 = resize_h(im3, H)
                                im4 = resize_h(im4, H)

                                concat5 = cv2.hconcat([im0, im1, im2, im3, im4])
                                sum_savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir,
                                                            f'{self.global_step:06}_sum.jpg')
                                cv2.imwrite(sum_savepath, concat5)
                        except Exception as e:
                            print(f"[sum-save] skip due to error: {e}")

                        # import ipdb; ipdb.set_trace()
                        # self.writer.add_image('train_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)

                        print("epoch and step:", epoch, step, self.global_step)

                if self.global_step > 0 and self.global_step % self.cfg.train.checkpoint_steps == 0 or step == (
                        iters_every_epoch - 1):
                    model_dict = self.mymodel.model_dict()
                    # model_dict = {key: model_dict[key]}
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))
                    # 
                    # if self.global_step % self.cfg.train.checkpoint_steps*10 == 0 or step == (iters_every_epoch-1):
                    if step == (iters_every_epoch - 1):
                        os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                        torch.save(model_dict,
                                   os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))
                        #
                # if self.global_step % self.cfg.train.val_steps == 0:
                #     self.validation_step()
                #
                # if self.global_step % self.cfg.train.eval_steps == 0:
                #     self.evaluate()

                all_loss = losses['all_loss']
                self.opt.zero_grad();
                all_loss.backward();
                self.opt.step();  # self.scheduler.step();
                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break