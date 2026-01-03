'''
디테일뉴(기본 )->디테일뉴브랜치(latent code 제거)->디코더 트레인
->v3 (코얼스 제거)-> onlyangry 2025 1228
'''
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
from .utils.config_wt_DetailNewBranch_v3 import cfg
from .utils import lossfuncN_DetailNew as lossfunc
from .utils.util import au_weights
from .models.expression_loss import ExpressionLossNet
from .models.OpenGraphAU.model.MEFL import MEFARG
from .models.OpenGraphAU.utils import load_state_dict
from .models.OpenGraphAU.utils import *
from .models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env
from .datasets import build_datasets_video_onlyangry as build_datasets
# from .datasets import build_datasets_image as build_datasets
# from .datasets import build_datasets_NoAug
torch.backends.cudnn.benchmark = True

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
        self.middleframe = self.cfg.dataset.K // 2
        # training stage: coarse and detail
        self.train_detail = self.cfg.train.train_detail
        self.train_coarse = self.cfg.train.train_coarse  # Added train_coarse flag
        self.vis_au = self.cfg.train.vis_au     #check line 471 ~486: 결과 코드에서 AU 모드로 출력할지.

        # mymodel model
        self.mymodel = model
        self.configure_optimizers()
        self.load_checkpoint()
        self.au_weight = au_weights()
        # if self.cfg.loss.weightedAU:
            # self.weightedAU = np.load('/mnt/hdd/EncoderTrainingCode/Code/data/AU_weight.npy')
            # self.weightedAU = torch.tensor(self.weightedAU).to(self.device)
            # self.au_weight = au_weights()

        # initialize loss
        if self.train_detail:
             self.mrf_loss = lossfunc.IDMRFLoss()
             self.face_attr_mask = util.load_local_mask(image_size=self.cfg.model.uv_size, mode='bbx')
        # else:
        #     self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)      
        
        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))
    
    def configure_optimizers(self):
        if self.train_coarse:
            self.opt_coarse = torch.optim.Adam(
                self.mymodel.BiViT.parameters(),
                lr=self.cfg.train.lr,
                amsgrad=False
            )
        else:
            self.opt_coarse = None
        
        if self.train_detail and hasattr(self.mymodel, 'ViTDetail'):
            self.opt_detail = torch.optim.Adam(
                list(self.mymodel.ViTDetail.parameters()) + \
                list(self.mymodel.D_detail.parameters()),
                lr=self.cfg.train.lr,
                amsgrad=False
            )
        else:
            self.opt_detail = None

    def load_checkpoint(self):
        # au config
        self.auconf = get_config()
        self.auconf.evaluate = True
        self.auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        set_env(self.auconf)
        model_dict = self.mymodel.model_dict()

        self.expression_net = ExpressionLossNet().to(self.device)
        emotion_checkpoint = torch.load(self.cfg.emotion_checkpoint, map_location='cpu')['state_dict']
        emotion_checkpoint['linear.0.weight'] = emotion_checkpoint['linear.weight']
        emotion_checkpoint['linear.0.bias'] = emotion_checkpoint['linear.bias']
        self.expression_net.load_state_dict(emotion_checkpoint, strict=False)
        self.expression_net.eval()
        self.AU_net = MEFARG(num_main_classes=self.auconf.num_main_classes, num_sub_classes=self.auconf.num_sub_classes, backbone=self.auconf.arc).to(self.device)
        self.AU_net = load_state_dict(self.AU_net, self.auconf.resume).to(self.device)
        self.AU_net.eval()
        
        # ========== 가중치 로딩 순서 (중요!) ==========
        # 1. pretrained_modelpath: 초기 가중치 로딩
        # 2. pretrained_coarse_modelpath: coarse 모듈 덮어쓰기
        # 3. Resume: trainable 모듈만 덮어쓰기 (학습된 가중치)
        
        # ========== 1단계: pretrained_modelpath에서 모든 모듈 초기화 ==========
        if os.path.exists(self.cfg.pretrained_modelpath):
            logger.info(f'[Step 1/3] Loading pretrained model weights from {self.cfg.pretrained_modelpath}')
            checkpoint = torch.load(self.cfg.pretrained_modelpath, map_location='cpu')
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
                    logger.info(f'  Loaded {key} from pretrained_modelpath')
            
            # Load D_detail_original from pretrained D_detail weights (frozen decoder)
            if 'D_detail' in checkpoint.keys() and hasattr(self.mymodel, 'D_detail_original'):
                util.copy_state_dict(self.mymodel.D_detail_original.state_dict(), checkpoint['D_detail'])
                self.mymodel.D_detail_original.eval()
                logger.info('  D_detail_original loaded from pretrained D_detail weights')
        else:
            logger.info('[Step 1/3] pretrained_modelpath not found, starting from scratch')
        
        # ========== 2단계: pretrained_coarse_modelpath에서 coarse 모듈 덮어쓰기 ==========
        if os.path.exists(self.cfg.pretrained_coarse_modelpath):
            logger.info(f'[Step 2/3] Loading coarse weights from pretrained_coarse_modelpath: {self.cfg.pretrained_coarse_modelpath}')
            logger.info('  This may take a while if the checkpoint file is large...')
            checkpoint_coarse = torch.load(self.cfg.pretrained_coarse_modelpath, map_location='cpu')
            logger.info('  Checkpoint loaded successfully, copying weights...')
            model_dict = self.mymodel.model_dict()
            
            # Load E_flame and BiViT from coarse checkpoint
            if 'E_flame' in checkpoint_coarse.keys():
                util.copy_state_dict(model_dict['E_flame'], checkpoint_coarse['E_flame'])
                logger.info('  E_flame loaded successfully from pretrained_coarse_modelpath')
            else:
                logger.warning('  E_flame not found in pretrained_coarse_modelpath checkpoint')
            
            if 'BiViT' in checkpoint_coarse.keys():
                util.copy_state_dict(model_dict['BiViT'], checkpoint_coarse['BiViT'])
                logger.info('  BiViT loaded successfully from pretrained_coarse_modelpath')
            else:
                logger.warning('  BiViT not found in pretrained_coarse_modelpath checkpoint')
            
            # Set train/freeze mode based on train_coarse flag
            if self.train_coarse:
                for param in self.mymodel.E_flame.parameters():
                    param.requires_grad = False  # E_flame is always frozen
                for param in self.mymodel.BiViT.parameters():
                    param.requires_grad = True  # BiViT will be trained
                logger.info('  Coarse weights loaded: BiViT will be fine-tuned (train_coarse=True)')
            else:
                for param in self.mymodel.E_flame.parameters():
                    param.requires_grad = False
                for param in self.mymodel.BiViT.parameters():
                    param.requires_grad = False
                self.mymodel.E_flame.eval()
                self.mymodel.BiViT.eval()
                logger.info('  Coarse weights loaded: E_flame and BiViT frozen (train_coarse=False)')
            logger.info('[Step 2/3] pretrained_coarse_modelpath loading completed')
        else:
            # Path not provided
            if self.train_coarse:
                logger.info('[Step 2/3] train_coarse=True but pretrained_coarse_modelpath not provided: coarse will train from scratch')
                for param in self.mymodel.BiViT.parameters():
                    param.requires_grad = True
                logger.info('  BiViT set to trainable mode (requires_grad=True) for training from scratch')
            else:
                logger.error('=' * 80)
                logger.error('ERROR: train_coarse=False but pretrained_coarse_modelpath is not provided!')
                logger.error(f'pretrained_coarse_modelpath: {self.cfg.pretrained_coarse_modelpath}')
                logger.error('When train_coarse=False, you MUST provide pretrained_coarse_modelpath to load coarse weights.')
                logger.error('Please set pretrained_coarse_modelpath in your config file (yml).')
                logger.error('=' * 80)
                raise ValueError('train_coarse=False requires pretrained_coarse_modelpath to be set. Please check your config file.')
        
        # ========== 3단계: Resume - trainable 모듈만 덮어쓰기 (학습된 가중치) ==========
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            logger.info(f'[Step 3/3] Resuming training from {os.path.join(self.cfg.output_dir, "model.tar")}')
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'), map_location='cpu')
            model_dict = self.mymodel.model_dict()
            
            # Trainable 모듈만 로딩 (pretrained 경로들을 덮어씀)
            # BiViT (train_coarse=True일 때만)
            if self.train_coarse and 'BiViT' in checkpoint.keys():
                util.copy_state_dict(model_dict['BiViT'], checkpoint['BiViT'])
                logger.info('  BiViT loaded from resume checkpoint (overwrites pretrained)')
            
            # ViTDetail (train_detail=True일 때만)
            if self.train_detail and 'ViTDetail' in checkpoint.keys():
                util.copy_state_dict(model_dict['ViTDetail'], checkpoint['ViTDetail'])
                logger.info('  ViTDetail loaded from resume checkpoint (overwrites pretrained)')
            
            # D_detail (train_detail=True일 때만)
            if self.train_detail and 'D_detail' in checkpoint.keys():
                util.copy_state_dict(model_dict['D_detail'], checkpoint['D_detail'])
                logger.info('  D_detail loaded from resume checkpoint (overwrites pretrained)')
            
            # Optimizers 로딩
            if self.opt_coarse is not None and 'opt_coarse' in checkpoint and checkpoint['opt_coarse'] is not None:
                util.copy_state_dict(self.opt_coarse.state_dict(), checkpoint['opt_coarse'])
                logger.info('  opt_coarse loaded from resume checkpoint')
            elif self.opt_coarse is None and 'opt_coarse' in checkpoint:
                logger.info('  opt_coarse found in checkpoint but train_coarse=False, skipping opt_coarse loading')
            
            if self.opt_detail is not None and 'opt_detail' in checkpoint and checkpoint['opt_detail'] is not None:
                util.copy_state_dict(self.opt_detail.state_dict(), checkpoint['opt_detail'])
                logger.info('  opt_detail loaded from resume checkpoint')
            
            # global_step 로딩
            self.global_step = checkpoint.get('global_step', 0)
            logger.info(f'[Step 3/3] Training will resume from step {self.global_step}')
        else:
            logger.info('[Step 3/3] Resume checkpoint not found, starting training from scratch')
            self.global_step = 0

    def training_step(self, batch, batch_nb, training_type='coarse'):
        # Performance optimization: Set individual module modes instead of self.mymodel.eval()
        # This allows trainable modules (ViTDetail, D_detail) to stay in train mode
        self.mymodel.E_flame.eval()
        self.mymodel.E_detail.eval()
        self.mymodel.AUNet.eval()
        
        # Set BiViT mode based on train_coarse flag
        if self.train_coarse:
            self.mymodel.BiViT.train()
        else:
            self.mymodel.BiViT.eval()  # Freeze when not training
        
        if self.train_detail and hasattr(self.mymodel, 'ViTDetail'):
            self.mymodel.ViTDetail.train()
            self.mymodel.D_detail.train()
            # Ensure D_detail_original stays in eval mode (frozen)
            if hasattr(self.mymodel, 'D_detail_original'):
                self.mymodel.D_detail_original.eval()

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images_224 = batch['image_224'].to(self.device); images_224 = images_224.view(-1, images_224.shape[-3], images_224.shape[-2], images_224.shape[-1])
        lmk = batch['landmark'].to(self.device); lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        lmk_dense = batch['landmark_dense'].to(self.device); lmk_dense = lmk_dense.view(-1, lmk_dense.shape[-2], lmk_dense.shape[-1])
        masks = batch['mask'].to(self.device); masks = masks.view(-1,images_224.shape[-3], images_224.shape[-2], images_224.shape[-1])
        
        #-- encoder
        # Performance optimization: Pass use_coarse_grad to skip BiViT gradient computation when train_coarse=False
        codedict_old, codedict = self.mymodel.encode(images_224, use_coarse_grad=self.train_coarse) #split image automaticalliy in encode
        # images = images_224
        images = images_224[self.middleframe:self.middleframe+1]
        lmk = lmk[self.middleframe:self.middleframe+1]
        lmk_dense = lmk_dense[self.middleframe:self.middleframe+1]
        masks = masks[self.middleframe:self.middleframe+1]
        #print(f"[DEBUG] masks[middleframe] 후 shape: {masks.shape}")

        masks_original = masks.clone()
        batch_size = 1          # later problem: batch size fix to 1. why????
        # batch_size = images_224.shape[0]  # ORIGINAL DECA.

        ###--------------- training coarse transformer + detail transformer
        #different loss function but same step.
        losses = {}
        
        # Coarse loss 계산은 train_coarse=True일 때만 수행
        if self.train_coarse:
            #-- decoder
            # rendering = True if self.cfg.loss.photo>0 else False
            rendering = True
            # codedict['tex'][:,:] = 2.0
            # opdict_old = self.mymodel.decode(codedict_old, rendering = rendering, vis_lmk=False, return_vis=False, use_detail=False)
            opdict = self.mymodel.decode(
                codedict, codedict_old, rendering = rendering,
                vis_lmk=False, return_vis=False, use_detail=False)
            opdict['images'] = images
            opdict['lmk'] = lmk
            opdict['lmk_dense'] = lmk_dense

            if self.cfg.loss.photo > 0.:
                #------ rendering
                # mask
                mask_face_eye = F.grid_sample(
                    self.mymodel.uv_face_eye_mask.to(opdict['grid'].device, non_blocking=True).expand(batch_size, -1, -1, -1),
                    opdict['grid'].detach(), align_corners=False
                )
                # images
                predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
                opdict['predicted_images'] = predicted_images

            #### ----------------------- Losses
            ############################# base shape
            predicted_landmarks = opdict['landmarks2d']
            predicted_landmarks_dense = opdict['mp_landmark']
            if self.cfg.loss.AUFLoss > 0 :
                _, afnT = self.mymodel.AUNet(images, use_gnn=False)
                _, afnP = self.mymodel.AUNet( opdict['rendered_images'], use_gnn=False)
                losses['AUFLoss'] = F.mse_loss(afnT.view(-1,1), afnP.view(-1,1)) * self.cfg.loss.expression

          # if self.cfg.loss.AUloss:
          #       image_au = self.AU_net(images)[1]
          #       rend_au = self.AU_net(opdict['rendered_images'])[1]
            if self.cfg.loss.weightedAU:
                # rend_au_loss = self.AU_net(opdict['rendered_images'])[1][0]*self.weightedAU
                image_au = self.AU_net(images)[1]
                rend_au = self.AU_net(opdict['rendered_images'])[1]
                losses['au_lmk_loss'] = lossfunc.weighted_au_landmark_loss(predicted_landmarks_dense, lmk_dense, image_au.float(), self.au_weight) * self.cfg.loss.mainAU
                losses['au_loss'] = F.mse_loss(image_au, rend_au)
                opdict['au_img'] = (image_au >= 0.5).float()
                opdict['au_rend'] = (rend_au >= 0.5).float()
            elif self.cfg.loss.focalAU:
                # if len(image_au) != self.batch_size:
                image_au = self.AU_net(images)
                rend_au = self.AU_net(opdict['rendered_images'])
                losses['au_lmk_loss'] = lossfunc.weighted_au_landmark_loss(predicted_landmarks_dense, lmk_dense, image_au[1].float(), self.au_weight) * self.cfg.loss.mainAU
                losses['au_dist_loss'] = lossfunc.related_au_landmark_loss(predicted_landmarks, lmk, self.au_weight) * self.cfg.loss.subAU
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
                    masks_coarse = masks[:, None, :, :]
                else:
                    masks_coarse = mask_face_eye * opdict['alpha_images']
                losses['photometric_texture'] = (masks_coarse * (
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


            if self.cfg.loss.reg_shape>0:
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
            #----------coarse end
        else:
            # train_coarse=False일 때는 coarse loss 계산 생략
            # Detail 학습에 필요한 최소한의 정보만 유지
            losses['all_loss'] = 0.0
            # opdict는 Detail 학습에 필요하므로 decode는 수행하되 loss 계산은 생략
            rendering = True
            opdict = self.mymodel.decode(
                codedict, codedict_old, rendering = rendering,
                vis_lmk=False, return_vis=False, use_detail=False)
            opdict['images'] = images
            opdict['lmk'] = lmk
            opdict['lmk_dense'] = lmk_dense

        ###------------------training detail model
        shapecode = codedict['shape']
        expcode = codedict['exp']
        posecode = codedict['pose']#
        texcode = codedict['tex']
        lightcode = codedict['light']
        detailcode = codedict['detail']
        cam = codedict['cam']

        #print(f"[DEBUG] codedict shapes:")
        #print(f"  shapecode: {shapecode.shape}")
        #print(f"  expcode: {expcode.shape}")
        #print(f"  posecode: {posecode.shape}")
        #print(f"  texcode: {texcode.shape}")
        #print(f"  lightcode: {lightcode.shape}")
        #print(f"  detailcode: {detailcode.shape}")
        #print(f"  cam: {cam.shape}")

        # FLAME 전방계산 (원문 동일)
        verts, landmarks2d, landmarks3d, _= self.mymodel.flame(shape_params=shapecode,
                                                 expression_params=expcode,
                                                 pose_params=posecode)
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:] #; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        
        
        predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:,:,:2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]
        # 디버깅: trans_verts shape
        #print(f"[DEBUG] trans_verts shape: {trans_verts.shape}")


        albedo = self.mymodel.flametex(texcode)
        ops = self.mymodel.render(verts, trans_verts, albedo, lightcode)
        
        masks_detail = masks_original[:,0:1,:,:]    #mask have same channel
        
        #make mymodel to make uv_z
        ##################rendr detail at trainer side####################
        # 원본의 코드와 다르게 D_detail에 들어가는 것의 형태가 좀 다른 것 같습니다.
        # Performance optimization: GPU 하나만 사용하므로 device 간 이동 제거

        cond_d = torch.cat([posecode[:, 3:], expcode, detailcode], dim=1)
        uv_z = self.mymodel.D_detail(cond_d)  # 같은 device에서 실행
     

        # render detail: displacement -> normal -> shading -> uv_texture (원문 동일)
        uv_detail_normals = self.mymodel.displacement2normal(uv_z, verts, ops['normals'])
        uv_shading = self.mymodel.render.add_SHlight(uv_detail_normals, lightcode.detach())
        uv_texture = albedo.detach() * uv_shading


        # detail 이미지를 uv grid로 샘플링 (원문 동일)
        predicted_detail_images = F.grid_sample(
            uv_texture, ops['grid'].detach(), align_corners=False
        )
        ####################render detail end####################

        # extract texture
        uv_pverts = self.mymodel.render.world2uv(trans_verts).detach()
                # 디버깅: uv_pverts shape 및 F.grid_sample 입력 shapes
        #print(f"[DEBUG] uv_pverts shape: {uv_pverts.shape}")
        #print(f"[DEBUG] uv_pverts.permute(0, 2, 3, 1) shape: {uv_pverts.permute(0, 2, 3, 1).shape}")
        #print(f"[DEBUG] uv_pverts.permute(0, 2, 3, 1)[..., :2] shape: {uv_pverts.permute(0, 2, 3, 1)[..., :2].shape}")
        #print(f"[DEBUG] torch.cat([images, masks_detail], dim=1) shape: {torch.cat([images, masks_detail], dim=1).shape}")
        
        
        uv_gt = F.grid_sample(torch.cat([images, masks_detail], dim=1),
                              uv_pverts.permute(0, 2, 3, 1)[..., :2],
                              mode='bilinear', align_corners=False)

        #print(f"[DEBUG] uv_gt shape: {uv_gt.shape}")
        uv_tex_gt = uv_gt[:, :3, :, :].detach()
        uv_mask_gt = uv_gt[:, 3:, :, :].detach()

        #self occlusion
        normals = util.vertex_normals(trans_verts, self.mymodel.render.faces.expand(batch_size, -1, -1))
        uv_pnorm = self.mymodel.render.world2uv(normals)
        uv_mask = (uv_pnorm[:, [-1], :, :] < -0.05).float().detach()
        #mask combine

        uv_vis_mask = uv_mask_gt * uv_mask * self.mymodel.uv_face_eye_mask

        #### ------ losses
        pi = 0
        new_size = 256
        # face_attr_mask는 __init__에서 util.load_local_mask(..., mode='bbx')로 준비되어 있어야 합니다.
        x0, x1, y0, y1 = self.face_attr_mask[pi]  # (left, right, top, bottom)
        uv_texture_patch = F.interpolate(uv_texture[:, :, y0:y1, x0:x1], [new_size, new_size], mode='bilinear')
        uv_texture_gt_patch = F.interpolate(uv_tex_gt[:, :, y0:y1, x0:x1], [new_size, new_size], mode='bilinear')
        uv_vis_mask_patch = F.interpolate(uv_vis_mask[:, :, y0:y1, x0:x1], [new_size, new_size], mode='bilinear')

        losses['photo_detail'] = (uv_texture_patch * uv_vis_mask_patch - uv_texture_gt_patch * uv_vis_mask_patch).abs().mean() * self.cfg.loss.photo_D
        losses['photo_detail_mrf'] = self.mrf_loss(uv_texture_patch * uv_vis_mask_patch,
                                                   uv_texture_gt_patch * uv_vis_mask_patch) * self.cfg.loss.photo_D * self.cfg.loss.mrf

        losses['z_reg'] = torch.mean(uv_z.abs())*self.cfg.loss.reg_z
        losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading)*self.cfg.loss.reg_diff
        if self.cfg.loss.reg_sym > 0.:
            nonvis_mask = (1 - util.binary_erosion(uv_vis_mask))
            losses['z_sym'] = (nonvis_mask*(uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum()*self.cfg.loss.reg_sym

        # detail 총합
        losses['all_loss_detail'] = (
                losses['photo_detail']
                + losses.get('photo_detail_mrf', 0.0)
                + losses['z_reg'] + losses['z_diff']
                + (losses.get('z_sym', 0.0))
        )

        # 시각화용 결과도 opdict에 추가(원문과 이름 호환)
        opdict['predicted_detail_images'] = predicted_detail_images
        opdict['trans_verts'] = trans_verts
        opdict['uv_texture'] = uv_texture
        opdict['uv_detail_normals'] = uv_detail_normals

        return losses, opdict

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

        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            # random.shuffle(self.train_dataset)
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.train.max_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                losses, opdict = self.training_step(batch, step)
                # time to log loss values!
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/'+k, v, global_step=self.global_step)                    
                    logger.info(loss_info)
                    
                    # time to render vertical result images!
                    if self.global_step % self.cfg.train.vis_steps == 0 or step == (iters_every_epoch-1):
                        # visind = list(range(self.cfg.dataset.batch_size* self.cfg.dataset.K)) #!!!!!!
                        visind = list(range(self.cfg.dataset.batch_size))

                        # render my images
                        shape_images_full, shape_images_face, _, grid, alpha_images = self.mymodel.render.render_shape(
                            opdict['verts'][visind], 
                            opdict['trans_verts'][visind], 
                            images=opdict['images'][visind],
                            return_grid=True
                            )
                
                        shape_images_full_old, shape_images_face_old = self.mymodel.render.render_shape(
                                opdict['verts_old'][visind], 
                                opdict['trans_verts_old'][visind],
                                images=opdict['images'][visind]
                            )
                        
                        ########## sshape detail add here
                        # shape_detail_images 생성 (회색 mesh with detail normal + original background)
                        shape_detail_images_full = None
                        shape_detail_images = None
                        shape_detail_images_full_old = None
                        shape_detail_images_old = None
                        
                        # DEBUG: opdict 키 확인
                        #print(f"[DEBUG] opdict keys related to detail: {[k for k in opdict.keys() if 'detail' in k or 'normal' in k]}")
                        
                        if 'uv_detail_normals' in opdict:
                            #print(f"[DEBUG] uv_detail_normals found! shape: {opdict['uv_detail_normals'].shape}")
                            # OUR detail: uv_detail_normals를 screen space로 변환
                            detail_normal_images = F.grid_sample(
                                opdict['uv_detail_normals'][visind], 
                                grid, 
                                align_corners=False
                            ) * alpha_images
                            
                            # OUR detail normal을 사용해서 회색 mesh 렌더링
                            shape_detail_images_full, shape_detail_images = self.mymodel.render.render_shape(
                                opdict['verts'][visind],
                                opdict['trans_verts'][visind],
                                detail_normal_images=detail_normal_images,
                                images=opdict['images'][visind]
                            )
                            #print(f"[DEBUG] OUR detail images created - shape_detail_images: {shape_detail_images.shape if shape_detail_images is not None else None}, shape_detail_images_full: {shape_detail_images_full.shape if shape_detail_images_full is not None else None}")
                        #else:
                            #print(f"[DEBUG] uv_detail_normals NOT found in opdict!")
                        
                        # DECA old detail: uv_detail_normals_old가 있으면 생성
                        if 'uv_detail_normals_old' in opdict:
                            #print(f"[DEBUG] uv_detail_normals_old found! shape: {opdict['uv_detail_normals_old'].shape}")
                            # old 버전용 grid와 alpha_images 생성
                            _, _, _, grid_old, alpha_images_old = self.mymodel.render.render_shape(
                                opdict['verts_old'][visind],
                                opdict['trans_verts_old'][visind],
                                images=opdict['images'][visind],
                                return_grid=True
                            )
                            
                            # uv_detail_normals_old를 screen space로 변환
                            detail_normal_images_old = F.grid_sample(
                                opdict['uv_detail_normals_old'][visind],
                                grid_old,
                                align_corners=False
                            ) * alpha_images_old
                            
                            # DECA old detail normal을 사용해서 회색 mesh 렌더링
                            shape_detail_images_full_old, shape_detail_images_old = self.mymodel.render.render_shape(
                                opdict['verts_old'][visind],
                                opdict['trans_verts_old'][visind],
                                detail_normal_images=detail_normal_images_old,
                                images=opdict['images'][visind]
                            )
                            #print(f"[DEBUG] OLD detail images created - shape_detail_images_old: {shape_detail_images_old.shape if shape_detail_images_old is not None else None}, shape_detail_images_full_old: {shape_detail_images_full_old.shape if shape_detail_images_full_old is not None else None}")
                        #else:
                            #print(f"[DEBUG] uv_detail_normals_old NOT found in opdict!")
                        ########## shape detail end here

                        
                        if self.vis_au:
                            visdict = {
                                'inputs': opdict['images'][visind],
                                # 'landmarks2d_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind], isScale=True),
                                # 'landmarks2d': util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind], isScale=True),
                                'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind], opdict['mp_landmark'][visind], opdict['lmk_dense'][visind], isScale=True),
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
                                'landmarks_dens_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk_dense'][visind], isScale=True),
                                'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind], opdict['mp_landmark'][visind], isScale=True),
                                'shape_images': shape_images_face,
                                'shape_images_full': shape_images_full,
                                'rendered_images': opdict['rendered_images']
                                # 'rendered_images': opdict['rendered_images']
                                # 'predicted_images': opdict['predicted_images'][visind],
                            }
                        # if 'predicted_images' in opdict.keys():
                        #     visdict['predicted_images'] = opdict['predicted_images'][visind]
                        if 'predicted_detail_images' in opdict.keys():
                            visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]

                        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}.jpg')
                        grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
                        
                        # detail 이미지들을 별도 파일로 저장 (왼쪽: 원본 + coarse, 오른쪽: 원본 + detail)
                        #print(f"[DEBUG] Checking shape_detail_images: {shape_detail_images is not None}")
                        if shape_detail_images is not None:
                            detail_savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}_detail.jpg')
                            #print(f"[DEBUG] Detail savepath: {detail_savepath}")
                            
                            # 왼쪽: 원본(맨 위) + coarse shape 이미지들
                            left_visdict = {}
                            left_visdict['inputs'] = opdict['images'][visind]
                            if shape_images_face is not None:
                                left_visdict['shape_images_face'] = shape_images_face
                            if shape_images_full is not None:
                                left_visdict['shape_images_full'] = shape_images_full
                            if shape_images_face_old is not None:
                                left_visdict['shape_images_face_old'] = shape_images_face_old
                            if shape_images_full_old is not None:
                                left_visdict['shape_images_full_old'] = shape_images_full_old
                            
                            # 오른쪽: albedo texture + detail(맨 위) + detail shape 이미지들
                            right_visdict = {}
                            # 맨 위에 albedo texture를 OUR detail+coarse에 적용한 결과
                            if 'predicted_detail_images' in opdict:
                                right_visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]
                            if shape_detail_images is not None:
                                right_visdict['shape_detail_images'] = shape_detail_images
                            if shape_detail_images_full is not None:
                                right_visdict['shape_detail_images_full'] = shape_detail_images_full
                            if shape_detail_images_old is not None:
                                right_visdict['shape_detail_images_old'] = shape_detail_images_old
                            if shape_detail_images_full_old is not None:
                                right_visdict['shape_detail_images_full_old'] = shape_detail_images_full_old
                            
                            # 왼쪽과 오른쪽 그리드 생성
                            left_grid = util.visualize_grid(left_visdict, savepath=None, size=224, dim=1, return_gird=True) if left_visdict else None
                            right_grid = util.visualize_grid(right_visdict, savepath=None, size=224, dim=1, return_gird=True) if right_visdict else None
                            
                            # 두 그리드를 가로로 연결
                            if left_grid is not None and right_grid is not None:
                                h_left, w_left = left_grid.shape[:2]
                                h_right, w_right = right_grid.shape[:2]
                                h = max(h_left, h_right)
                                
                                # 높이 맞추기 (패딩)
                                if h_left < h:
                                    pad = (h - h_left) // 2
                                    left_grid = np.pad(left_grid, ((pad, h-h_left-pad), (0, 0), (0, 0)), mode='constant', constant_values=255)
                                if h_right < h:
                                    pad = (h - h_right) // 2
                                    right_grid = np.pad(right_grid, ((pad, h-h_right-pad), (0, 0), (0, 0)), mode='constant', constant_values=255)
                                
                                grid_image = np.concatenate([left_grid, right_grid], axis=1)
                                cv2.imwrite(detail_savepath, grid_image)
                            elif left_grid is not None:
                                cv2.imwrite(detail_savepath, left_grid)
                            elif right_grid is not None:
                                cv2.imwrite(detail_savepath, right_grid)
                            #else:
                                #print(f"[DEBUG] detail_visdict is empty, not saving detail file")
                        #else:
                            #print(f"[DEBUG] shape_detail_images is None, skipping detail file save")
                        
                        # import ipdb; ipdb.set_trace()                    
                        # self.writer.add_image('train_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)

                        print("epoch and step:", epoch, step, self.global_step)
                    
                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0 or step == (iters_every_epoch-1):
                    model_dict = self.mymodel.model_dict()
                    # model_dict = {key: model_dict[key]}

                    # Save optimizers only if they exist
                    if self.opt_coarse is not None:
                        model_dict['opt_coarse'] = self.opt_coarse.state_dict()
                    else:
                        model_dict['opt_coarse'] = None  # Explicitly save None
                    
                    if self.opt_detail is not None:
                        model_dict['opt_detail'] = self.opt_detail.state_dict()
                    else:
                        model_dict['opt_detail'] = None  # Explicitly save None

                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))
                    #
                    # if self.global_step % self.cfg.train.checkpoint_steps*10 == 0 or step == (iters_every_epoch-1):
                    if step == (iters_every_epoch-1):
                        os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                        torch.save(model_dict, os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))   
                #
                # if self.global_step % self.cfg.train.val_steps == 0:
                #     self.validation_step()
                #
                # if self.global_step % self.cfg.train.eval_steps == 0:
                #     self.evaluate()

                # Zero gradients only for active optimizers
                if self.opt_coarse is not None:
                    self.opt_coarse.zero_grad(set_to_none=True)
                if self.opt_detail is not None:
                    self.opt_detail.zero_grad(set_to_none=True)

                # coarse / detail 합계는 각각 losses['all_loss'], losses['all_loss_detail']로 계산되어 있음
                loss_c = losses.get('all_loss', 0.0)
                loss_d = losses.get('all_loss_detail', 0.0)

                # Backward and step only for active optimizers
                # Performance optimization: train_coarse=False일 때는 loss_c=0이므로 backward 생략
                if loss_d != 0.0:
                    if loss_c != 0.0:
                        # Both coarse and detail losses
                        torch.autograd.backward([loss_c, loss_d])
                        if self.opt_coarse is not None:
                            self.opt_coarse.step()
                        if self.opt_detail is not None:
                            self.opt_detail.step()
                    else:
                        # Only detail loss (train_coarse=False)
                        loss_d.backward()
                        if self.opt_detail is not None:
                            self.opt_detail.step()
                else:
                    # Only coarse loss (train_detail=False)
                    if loss_c != 0.0 and self.opt_coarse is not None:
                        loss_c.backward()
                        self.opt_coarse.step()

                self.global_step += 1
