import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from skimage.io import imread
# from .utils.renderer import SRenderY, set_rasterizer
from .utils.renderer3 import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from .models.encoders_au import AUEncoder
from .models.encoders_au_v2 import AUEncoder2
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .models.enc_net import Resnet50Encoder, Resnet50Encoder_v2
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.tensor_cropper import transform_points
from .utils.config_wt import cfg
from .utils.util import au_weights
from .models.OpenGraphAU.model.ANFL import AFG
from .models.GATE import GATE
from .models.encoders_au import AUEncoder
# from .models.TransformerDecoder import FLAMETransformerDecoderHead
from .models.OpenGraphAU.utils import load_state_dict
from .models.OpenGraphAU.utils import *
from .models.OpenGraphAU.conf import get_config, set_logger, set_outdir, set_env
from .models.vitVideo import ViTEncoderV
from .models.vitVideo_Sequence_Cross_WT_detail import ViTEncoderSequeMulti


# from .models.gat import GAT

def _finite(name, t):
    import torch
    assert torch.isfinite(t).all(), f"[NaNGuard] {name} has non-finite values"


torch.backends.cudnn.benchmark = True
from copy import deepcopy


# PerspectiveCameras.transform_points_screen()
class DECA(nn.Module):
    def __init__(self, config=None, device='cuda:1'):
        super(DECA, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.flame_mask_file = self.cfg.model.flame_mask_path
        self.batchsize = self.cfg.dataset.batch_size
        self.au_weight = au_weights()
        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)
        # self.n_scenelight = self.cfg.model.n_scenelight
        # self.n_facelight = self.cfg.model.n_facelight

    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        self.render = SRenderY(self.image_size, flame_mask_file=self.flame_mask_file,
                               obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
                               rasterizer_type=self.cfg.rasterizer_type).to(self.device)
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.;
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32) / 255.;
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32) / 255.;
        mean_texture = torch.from_numpy(mean_texture.transpose(2, 0, 1))[None, :, :, :].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

        self.lightprobe_normal_images = F.interpolate(
            torch.from_numpy(np.load(model_cfg.lightprobe_normal_path)).float(),
            [model_cfg.image_size, model_cfg.image_size]).to(self.device)
        self.lightprobe_albedo_images = F.interpolate(
            torch.from_numpy(np.load(model_cfg.lightprobe_albedo_path)).float(),
            [model_cfg.image_size, model_cfg.image_size]).to(self.device)

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = (model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose +
                        model_cfg.n_cam + model_cfg.n_light + model_cfg.n_shape)
        # self.n_param_part =  model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_light
        self.n_param_part = model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_light
        self.n_movem = model_cfg.n_exp + 3  # + 27  # exp + jaw pose + action unit
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}
        self.param_dict_OnlyE = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list_OnlyE}
        # self.param_dict['tex'] = 50
        self.n_param_OnlyE = sum(self.param_dict_OnlyE.values())
        self.n_detail = getattr(model_cfg, 'n_detail', 256)

        # au config
        self.auconf = get_config()
        self.auconf.evaluate = True
        self.auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        set_env(self.auconf)

        # encoders
        # self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
        self.BiViT = ViTEncoderSequeMulti(num_features_g=self.n_param_OnlyE,  # coarse 파라미터 총합
                                          num_features_d=self.n_detail)  # detail latent 크기

        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # self.E_detail_deca = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # self.D_detail_deca = Generator(latent_dim=self.n_detail + self.n_movem, out_channels=1, out_scale=model_cfg.max_z,
        #                           sample_mode='bilinear').to(self.device)
        #                           sample_mode='bilinear').to(self.device)

        # self.E_albedo = Resnet50Encoder_v2(outsize=model_cfg.n_tex).to(self.device)
        # self.E_scene_light = Resnet50Encoder(outsize=model_cfg.n_scenelight).to(self.device)
        # self.E_face_light = Resnet50Encoder(outsize=model_cfg.n_facelight).to(self.device)
        # scene_model_path = model_cfg.pretrained_modelpath_scene
        # facel_model_path = model_cfg.pretrained_modelpath_facel
        # albedo_model_path = model_cfg.pretrained_modelpath_albedo

        self.AUNet = AFG(num_main_classes=self.auconf.num_main_classes, num_sub_classes=self.auconf.num_sub_classes,
                         backbone=self.auconf.arc).to(self.device)
        # self.AUD_Encoder = AUEncoder2().to(self.device)
        # self.GATE = GATE(nfeat=512,
        #         nhid=512,
        #         noutput=59,
        #         dropout=0.0,
        #         nheads=4,
        #         alpha=0.2,
        #         batchsize=self.batchsize).to(self.device)
        # # self.GATE_detail = GATE(nfeat=512,
        #         nhid=512,
        #         noutput=128,
        #         dropout=0.0,
        #         nheads=1,
        #         alpha=0.2,
        #         batchsize=self.batchsize).to(self.device)
        # # self.FTD = FLAMETransformerDecoderHead().to(self.device)

        self.AUNet = load_state_dict(self.AUNet, self.auconf.resume).to(self.device)
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail + self.n_movem, out_channels=1, out_scale=model_cfg.max_z,
                                  sample_mode='bilinear').to(self.device)
        # resume modelwi
        model_path = self.cfg.pretrained_modelpath
        model_path_224 = self.cfg.pretrained_modelpath_224
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            # print(checkpoint.keys())
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.BiViT.state_dict(), checkpoint['BiViT'])
            parameters = self.E_flame.state_dict()
            # for name in self.E_flame.state_dict():
            #     print(name)
            self.weight = parameters['layers.0.weight']
            self.bias = parameters['layers.0.bias']
        elif os.path.exists(model_path_224):
            print(f'trained model found. load {model_path_224}')
            checkpoint = torch.load(model_path_224)
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            parameters = self.E_flame.state_dict()
            # for name in self.E_flame.state_dict():
            #     print(name)
            self.weight = parameters['layers.0.weight']
            self.bias = parameters['layers.0.bias']
            # print(parameters['layers.0.weight'].shape)
            # print(parameters['layers.0.bias'].shape)
            # print(parameters['layers.2.weight'].shape)
            # print(parameters['layers.2.bias'].shape)
        else:
            print(f'please check pretrained deca model path: {model_path_224}')
            exit()
        # eval mode

        # self.E_flame.eval()
        self.E_flame.eval()
        self.BiViT.eval()
        self.AUNet.eval()

    def decompose_code(self, code, num_dict):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        # code_list = []
        # num_list = [self.cfg.model.n_shape, 50, self.cfg.model.n_exp, self.cfg.model.n_pose, self.cfg.model.n_cam, self.cfg.model.n_light]
        # start = 0
        # for i in range(len(num_list)):
        #     code_list.append(code[:, start:start+num_list[i]])
        #     start = start + num_list[i]
        # code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        # return code_list
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        # num_list = [self.cfg.model.n_shape, 50, self.cfg.model.n_exp, self.cfg.model.n_pose, self.cfg.model.n_cam, self.cfg.model.n_light]
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            # [NaNGuard] 각 파트 분해 직후 확인
            _finite(f"decompose_code.slice.{key}", code_dict[key])
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
                _finite("decompose_code.slice.light.reshaped", code_dict[key])

        return code_dict

    def decompose_code_part(self, code, num_dict):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        # code_list = []
        # num_list = [self.cfg.model.n_shape, 50, self.cfg.model.n_exp, self.cfg.model.n_pose, self.cfg.model.n_cam, self.cfg.model.n_light]
        # start = 0
        # for i in range(len(num_list)):
        #     code_list.append(code[:, start:start+num_list[i]])
        #     start = start + num_list[i]
        # code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        # return code_list
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = [ 'tex', 'exp', 'jaw_pose', 'light']
        '''
        code_dict = {}
        # num_list = [self.cfg.model.n_shape, 50, self.cfg.model.n_exp, self.cfg.model.n_pose, self.cfg.model.n_cam, self.cfg.model.n_light]
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)

        return code_dict

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()

        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals + self.fixed_uv_dis[None, None, :,
                                                                             :] * uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        uv_detail_normals = uv_detail_normals * self.uv_face_eye_mask + uv_coarse_normals * (1. - self.uv_face_eye_mask)
        return uv_detail_normals

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:, :, 2:] < 0.1).float()
        return vis68

    from torch.nn import functional as F
    # @torch.no_grad()
    def encode(self, images_224):

        with torch.no_grad():
            parameters_224, global_feature = self.E_flame(images_224)

            # [NaNGuard] 입력과 E_flame 출력 확인
            _finite("encode.images_224", images_224)
            _finite("encode.parameters_224", parameters_224)

            _, afn = self.AUNet(images_224, use_gnn=False)
            out_detail = self.E_detail(images_224)
            detail_seq = out_detail[0] if isinstance(out_detail, tuple) else out_detail  # tensor change!
        # feats = torch.concat([afn, global_feature], dim=1)

        # _, n, s = feats.shape
        # global_feature = F.linear(global_feature, self.weight, self.bias)
        afn_flat = afn.view(afn.shape[0], -1)  # (T, au_in_dim)
        parameters_ours, detail_latent_new = self.BiViT(afn_flat, detail_seq, global_feature)
        # parameters_ours = self.ViT_LC(feats)
        if parameters_ours.shape[0] > 1:
            parameters_ours = torch.unsqueeze(parameters_ours, 0)
        parameters_224 = parameters_224[1:2]

        codedict_our = self.decompose_code_part(parameters_ours, self.param_dict_OnlyE)
        codedict_our['detail'] = detail_latent_new
        codedict = self.decompose_code(parameters_224, self.param_dict)

        # [NaNGuard] decompose 결과 각 파트 확인(특히 tex)
        _finite("encode.decompose.shape", codedict['shape'])
        _finite("encode.decompose.tex", codedict['tex'])
        _finite("encode.decompose.exp", codedict['exp'])
        _finite("encode.decompose.pose", codedict['pose'])
        _finite("encode.decompose.cam", codedict['cam'])
        _finite("encode.decompose.light", codedict['light'])

        codedict['images'] = images_224[1:2]
        codedict_our['images'] = images_224[1:2]
        codedict_our['shape'] = codedict['shape']
        # codedict_our['tex'] = codedict['tex']
        codedict_our['cam'] = codedict['cam']
        codedict_our['pose'] = deepcopy(codedict['pose'])
        # codedict_our['pose'][:,3:]=codedict_our['jaw_pose']

        return codedict, codedict_our

    # @torch.no_grad()
    def decode(self, codedict, codedict_old, rendering=True, iddict=None, vis_lmk=True, vis_au=False, return_vis=True,
               use_detail=True, use_gnn=False,
               render_orig=False, original_image=None, tform=None):
        images = codedict['images']
        # B, C, H, W = images.size()
        # batch_size = images.shape[0]

        ## decode
        verts, landmarks2d, landmarks3d, mp_landmark = self.flame(shape_params=codedict['shape'],
                                                                  expression_params=codedict['exp'],
                                                                  pose_params=codedict['pose'])
        verts_old, _, _, _ = self.flame(shape_params=codedict_old['shape'], expression_params=codedict_old['exp'],
                                        pose_params=codedict_old['pose'])
        # albedo = self.flametex(codedict['tex'])

        # codedict 생성 직후 (278 줄에서 만들어짐)
        _finite("codedict.tex", codedict['tex'])
        _finite("codedict.light", codedict['light'])  # light가 (B,9,3)로 reshape된 뒤임
        _finite("codedict.exp", codedict['exp'])
        _finite("codedict.pose", codedict['pose'])

        albedo = self.flametex(codedict['tex'])

        # albedo 계산 직후
        _finite("albedo", albedo)

        # if self.cfg.model.use_tex and self.cfg.model.tex_type == 'BalanceAlb':
        #     lightcode, scale_factors, normalized_sh_params = self.fuse_light(codedict['S_light'], codedict['F_light'])
        #     codedict['light'] = lightcode
        #     # print(codedict['light'])
        #     images = images.contiguous().view(B, C, H, W)
        #     scale_factors_img = scale_factors.contiguous().view(B, scale_factors.size(1), 1, 1).repeat(1, 1, H, W)
        #     images_cond = torch.cat((scale_factors_img, images), dim=1)
        #     texcode = self.E_albedo(images_cond)
        #     codedict['tex'] = texcode
        #     albedo = self.flametex(texcode)
        #
        # elif self.cfg.model.use_tex:
        #     albedo = self.flametex(codedict['tex'])
        # else:
        #     albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device)
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:, :, :2];
        landmarks2d[:, :, 1:] = -landmarks2d[:, :,
                                 1:]  # ; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']);
        landmarks3d[:, :, 1:] = -landmarks3d[:, :,
                                 1:]  # ; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        mp_landmark = util.batch_orth_proj(mp_landmark, codedict['cam'])[:, :, :2];

        mp_landmark[:, :, 1:] = -mp_landmark[:, :, 1:]

        trans_verts = util.batch_orth_proj(verts, codedict['cam']);
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        trans_verts_old = util.batch_orth_proj(verts_old, codedict_old['cam']);
        trans_verts_old[:, :, 1:] = -trans_verts_old[:, :, 1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'verts_old': verts_old,
            'trans_verts_old': trans_verts_old,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'mp_landmark': mp_landmark,
            'landmarks3d_world': landmarks3d_world,
        }

        ## rendering
        if return_vis and render_orig and original_image is not None and tform is not None:
            points_scale = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            # import ipdb; ipdb.set_trace()
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])
            landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])

            landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
            mp_landmark = transform_points(mp_landmark, tform, points_scale, [h, w])
            background = original_image
            images = original_image
        else:
            h, w = self.image_size, self.image_size
            # background = None
            background = images

        if rendering:
            # ops = self.render(verts, trans_verts, albedo, codedict['light'])
            ops = self.render(verts, trans_verts, albedo, codedict['light'], h=h, w=w)  # , background=background)
            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
            opdict['v_colors'] = ops['v_colors']

            # predicted_images_alpha = ops['images'] * ops['alpha_images']
            # predicted_albedo_images = ops['albedo_images'] * ops['alpha_images']
            # predicted_shading = self.lightprobe_shading(self.SH_convert(lightcode))

            # detail map을 이미지 공간으로 역투영하고, 입력과 가시성 마스크로 합성 ---
            # --- detail 합성 블럭: DECA와 동일한 의미로 동작 + 안전한 fallback ---
            if use_detail:
                have_uv = ('uv_texture' in opdict)
                have_grid = ('grid' in opdict)
                have_alpha = ('v_colors' in opdict)

                if have_uv and have_grid:
                    # 1) UV 텍스처(알베도*셰이딩)를 image space로 역투영
                    predicted_detail_images = F.grid_sample(
                        opdict['uv_texture'],  # [B,3,Huv,Wuv]
                        opdict['grid'].detach(),  # [B,H,W,2]
                        align_corners=False,
                        mode='bilinear',
                        padding_mode='zeros'
                    )

                    # 2) 가시성/알파(v_colors)로 입력 이미지와 블렌딩
                    #    ※ base 이미지는 decode() 스코프의 'images'를 사용 (opdict['images'] 의존 제거)
                    if have_alpha:
                        alpha = opdict['v_colors']
                        if alpha.dim() == 3:  # [B,H,W] -> [B,1,H,W]
                            alpha = alpha.unsqueeze(1)
                        render_images_detail = images * (1.0 - alpha) + predicted_detail_images * alpha
                    else:
                        # 알파가 없으면 디테일만 출력 (필요 시 coarse 복사 가능)
                        render_images_detail = predicted_detail_images

                    # 3) 시각화/후속 단계에서 쓸 수 있도록 저장
                    opdict['predicted_detail_images'] = predicted_detail_images
                    opdict['render_images_detail'] = render_images_detail

                else:
                    # --- Fallback: 샘플링 불가 시 최소한 coarse를 흘려보냄 ---
                    if 'rendered_images' in opdict:
                        opdict['predicted_detail_images'] = opdict['rendered_images']
                        opdict['render_images_detail'] = opdict['rendered_images']

        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo

        # ---------------- detail 분기 추가 ----------------
        if use_detail:
            # jaw pose(3), expression, detail latent을 하나의 조건 벡터로 만듭니다
            jaw = codedict['pose'][:, 3:6]  # (B,3)
            exp = codedict['exp']  # (B,n_exp)
            det = codedict.get('detail', None)  # Transformer에서 얻은 detail latent
            if det is None:
                det = torch.zeros((exp.shape[0], self.n_detail), device=exp.device)
            cond = torch.cat([jaw, exp, det], dim=1)

            # iddict가 주어졌다면 그쪽 jaw/exp를 사용
            if iddict is not None:
                cond = torch.cat([iddict['pose'][:, 3:6], iddict['exp'], det], dim=1)

            # detail decoder로 UV displacement map 생성
            _finite("detail.cond", cond)
            uv_z = self.D_detail(cond)
            _finite("detail.uv_z", uv_z)

            # 디스플레이스먼트맵을 UV 노멀 맵으로 변환
            _finite("render.coarse_normals", ops['normals'])
            uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
            _finite("detail.uv_detail_normals", uv_detail_normals)

            # 조명계수와 곱해 쉐이딩 계산
            _finite("codedict.light", codedict['light'])  # 중복이지만, 여기서도 다시 체크
            uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
            opdict['uv_shading'] = uv_shading  # detail-only(알베도 없는) 시각화를 위해 보관

            _finite("detail.uv_shading", uv_shading)

            # 알베도와 곱해 디테일 UV 텍스처 생성
            _finite("albedo(pre-uv_texture)", albedo)
            uv_texture = albedo * uv_shading
            _finite("detail.uv_texture", uv_texture)

            # 결과들을 opdict에 저장
            opdict['uv_texture'] = uv_texture
            opdict['normals'] = ops['normals']
            opdict['uv_detail_normals'] = uv_detail_normals
            # 고정 보정량을 더한 디스플레이스먼트맵
            opdict['displacement_map'] = uv_z + self.fixed_uv_dis[None, None, :, :]
        # -------------------------------------------------

        # if use_detail:
        #     uv_z = self.D_detail(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1))
        #     # uv_z = self.D_detail(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail'], codedict['au']], dim=1))
        #     if iddict is not None:
        #         uv_z = self.D_detail(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1))
        #         # uv_z = self.D_detail(torch.cat([iddict['pose'][:, 3:], iddict['exp'], codedict['detail'], codedict['au']], dim=1))
        #     uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
        #     uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
        #     uv_texture = albedo * uv_shading
        #
        #     opdict['uv_texture'] = uv_texture
        #     opdict['normals'] = ops['normals']
        #     opdict['uv_detail_normals'] = uv_detail_normals
        #     opdict['displacement_map'] = uv_z + self.fixed_uv_dis[None, None, :, :]

        if vis_lmk:
            landmarks3d_vis = self.visofp(ops['transformed_normals'])  # /self.image_size
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict['landmarks3d'] = landmarks3d

        if return_vis:
            ## render shape
            # shape_images = self.render.render_shape(verts, trans_verts)
            shape_images_full, shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h,
                                                                                              w=w,
                                                                                              images=background,
                                                                                              return_grid=True)
            shape_images_full_old, shape_images_old, _, _, _ = self.render.render_shape(verts_old, trans_verts_old, h=h,
                                                                                        w=w,
                                                                                        images=background,
                                                                                        return_grid=True)

            if use_detail:
                 predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)

                 detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False) * alpha_images
                 shape_detail_images_full, shape_detail_images = self.render.render_shape(verts, trans_verts,
                                                             detail_normal_images=detail_normal_images, h=h, w=w,
                                                             images=background)

            # ▼ opdict로 넘겨서 트레이너에서 그대로 쓰게
            opdict['shape_detail_images'] = shape_detail_images  # 디테일+코어스 메쉬(회색)
            opdict['shape_detail_images_full'] = shape_detail_images_full
            opdict['grid'] = grid
            opdict['alpha_images'] = alpha_images

            ## extract texture
            ## TODO: current resolution 256x256, support higher resolution, and add visibility
            uv_pverts = self.render.world2uv(trans_verts)
            uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear',
                                  align_corners=False)
            # if self.cfg.model.use_tex:
            #     ## TODO: poisson blending should give better-looking results
            #     if self.cfg.model.extract_tex:
            #         uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
            #                     uv_texture[:, :3, :, :] * (1 - self.uv_face_eye_mask))
            #     else:
            #         # uv_texture_gt = uv_texture[:, :3, :, :]
            #         uv_texture_gt = albedo[:, :3, :, :]
            # else:
            #     uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
            #                 torch.ones_like(uv_gt[:, :3, :, :]) * (1 - self.uv_face_eye_mask) * 0.7)
            #
            # opdict['uv_texture_gt'] = uv_texture_gt
            # if use_detail:
            #     visdict = {
            #         'inputs': images,
            #         # 'predicted_detail_images': predicted_detail_images,
            #         # 'predicted_images_alpha': predicted_images_alpha,
            #         # 'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
            #         # 'mp_landmark': util.tensor_vis_landmarks(images, mp_landmark),
            #         # 'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
            #         # 'shape_images': shape_images,
            #         # 'render_images2': opdict['rendered_images'],
            #         'shape_detail_images': shape_detail_images,
            #         'shape_images': shape_images,
            #         'render_images': images*(1-opdict['v_colors'])+predicted_detail_images*opdict['v_colors'],#opdict['rendered_images'],
            #     }
            # else:

            visdict = {
                'inputs': images,
                # 'predicted_images_alpha': predicted_images_alpha,
                # 'predicted_albedo_images': predicted_albedo_images,
                # 'predicted_shading': predicted_shading,
                # 'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                # 'mp_landmark': util.tensor_vis_landmarks(images, mp_landmark),
                # 'au_gt': util.draw_activation_circles(images, landmarks3d, codedict['au'], self.au_weight),
                # 'au_pred': util.draw_activation_circles(opdict['rendered_images'], opdict['landmarks3d'], opdict['au_rend'], self.au_weight),
                # 'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                'shape_images_old': shape_images_old,
                'shape_images_full_old': shape_images_full_old,
                'shape_images': shape_images,
                'shape_images_full': shape_images_full,
                'render_images': images * (1 - opdict['v_colors']) + opdict['rendered_images'] * opdict['v_colors'],
                # opdict['rendered_images'],
                # 'render_images2': opdict['rendered_images'],
                # 'shape_detail_images': shape_detail_images
            }
            # --- [ADD] detail 결과를 visdict에도 실어주기 ---
            if 'predicted_detail_images' in opdict:
                visdict['predicted_detail_images'] = opdict['predicted_detail_images']
            if 'render_images_detail' in opdict:
                # detail overlay가 적용된 완성 이미지
                visdict['render_images_detail'] = opdict['render_images_detail']

            # if self.cfg.model.use_tex:
            #     visdict['uv_texture_gt'] = uv_texture_gt #!!!
            # visdict['rendered_images'] = ops['images']

            return opdict, visdict

        else:
            return opdict

    # @torch.no_grad()

    def visualize(self, visdict, size=224, dim=2):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim == 2
        grids = {}
        for key in visdict:
            _, _, h, w = visdict[key].shape
            if dim == 2:
                new_h = size;
                new_w = int(w * size / h)
            elif dim == 1:
                new_h = int(h * size / w);
                new_w = size
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu())

        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

        return grid_image

    def SH_normalization(self, sh_param):
        norm_sh_param = F.normalize(sh_param, p=1, dim=1)
        return norm_sh_param

    def lightprobe_shading(self, sh):
        bz = sh.shape[0]
        lightprobe_normal = self.lightprobe_normal_images.expand(bz, -1, -1, -1)
        shading_image = self.render.add_SHlight(lightprobe_normal, sh)
        return shading_image

    def SH_convert(self, sh):
        '''
        rotate SH with pi around x axis
        '''
        gt_sh_inverted = sh.clone()
        gt_sh_inverted[:, 1, :] *= -1
        gt_sh_inverted[:, 2, :] *= -1
        gt_sh_inverted[:, 4, :] *= -1
        gt_sh_inverted[:, 7, :] *= -1

        return gt_sh_inverted

    def fuse_light(self, E_scene_light_pred, E_face_light_pred):
        normalized_sh_params = self.SH_normalization(E_face_light_pred)
        lightcode = E_scene_light_pred.unsqueeze(1).expand(-1, 9, -1) * normalized_sh_params

        return lightcode, E_scene_light_pred, normalized_sh_params

    # def save_obj(self, filename, opdict):
    #     '''
    #     vertices: [nv, 3], tensor
    #     texture: [3, h, w], tensor
    #     '''
    #     i = 0
    #     vertices = opdict['verts'][i].cpu().numpy()
    #     faces = self.render.faces[0].cpu().numpy()
    #     # texture = util.tensor2image(opdict['uv_texture_gt'][i])
    #     uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
    #     uvfaces = self.render.uvfaces[0].cpu().numpy()
    #     # save coarse mesh, with texture and normal map
    #     normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
    #     # print(opdict['normals'].shape)
    #     # normal_map = util.tensor2image(opdict['normals'][i] * 0.5 + 0.5)
    #     # normal_map = None
    #     util.write_obj(filename, vertices, faces,
    #                    # texture=texture,
    #                    uvcoords=uvcoords,
    #                    uvfaces=uvfaces,
    #                    normal_map=normal_map)from .utils.util import au_weights
    #     #                dense_faces,
    #     #                colors=dense_colors,
    #     #                inverse_face_order=True)
    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
        # normal_map = util.tensor2image(opdict['normals'][i] * 0.5 + 0.5)
        # normal_map = None
        util.write_obj(filename, vertices, faces,
                       texture=texture,
                       uvcoords=uvcoords,
                       uvfaces=uvfaces,
                       normal_map=normal_map)
        # upsample mesh, save detailed mesh
        # texture = texture[:, :, [2, 1, 0]]
        # normals = opdict['normals'][i].cpu().numpy()
        # displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
        # dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map,
        #                                                                texture, self.dense_template)
        # util.write_obj(filename.replace('.obj', '_detail.obj'),
        #                dense_vertices,
        #                dense_faces,
        #                colors=dense_colors,
        #                inverse_face_order=True)

    # def run(self, imagepath, iscrop=True):
    #     ''' An api for running deca given an image path
    #     '''
    #     testdata = datasets.TestData(imagepath)
    #     images = testdata[0]['image'].to(self.device)[None, ...]
    #     images_224 = testdata[0]['image_224'].to(self.device)[None, ...]
    #     codedict = self.encode(images, images_224)
    #     opdict, visdict = self.decode(codedict)
    #     return codedict, opdict, visdict

    def model_dict(self):
        return {
            # 'GATE': self.GATE.state_dict(),
            # 'FTD': self.FTD.state_dict(),
            # 'AU_Encoder': self.AU_Encoder.state_dict(),
            # 'GATE_detail': self.GATE_detail.state_dict(),
            # 'AUD_Encoder': self.AUD_Encoder.state_dict(),
            # 'AU_Net': self.AUNet.state_dict(),
            'E_flame': self.E_flame.state_dict(),
            # 'E_detail': self.E_detail.state_dict(),
            # 'D_detail': self.D_detail.state_dict()
            'BiViT': self.BiViT.state_dict(),
        }
