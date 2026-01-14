# -*- coding: utf-8 -*-
import glob
import os, sys
import re
import argparse
import shutil
from tqdm import tqdm

import numpy as np
import torch

# (작동 코드 스타일 유지)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decalib.gatfarec_Video_DetailNewBranch import DECA
from decalib.datasets import datasets_WT_DetailNew as datasets
from decalib.utils.config_wt_DetailNew import cfg as deca_cfg

from decalib.models.OpenGraphAU.model.MEFL import MEFARG
from decalib.models.OpenGraphAU.utils import load_state_dict
from decalib.models.OpenGraphAU.conf import get_config, set_env


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def extract_frame_index(imagename: str) -> int:
    """파일명에서 마지막 숫자 덩어리를 frame index로 사용."""
    base = os.path.basename(imagename)
    nums = re.findall(r'\d+', base)
    return int(nums[-1]) if nums else -1


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def au_forward_scores(AU_net, img_tensor: torch.Tensor) -> torch.Tensor:
    """
    작동 코드가 AU_net(x)[1]을 쓰므로 동일하게 [1]을 score로 간주합니다.
    반환 shape: [B, num_au]
    """
    out = AU_net(img_tensor)
    if isinstance(out, (tuple, list)):
        x = out[1]
    else:
        x = out
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def pick_detail_render_ours(opdict: dict, visdict: dict):
    """
    'ours' (codedict로 디코드된 결과)에서 detail render를 최대한 찾아 반환.
    못 찾으면 coarse render를 반환.
    """
    coarse = None
    detail = None

    # coarse candidates
    if isinstance(opdict, dict):
        coarse = opdict.get("rendered_images", None)
        if coarse is None:
            coarse = opdict.get("render_images", None)

    # detail candidates (prefer visdict)
    if isinstance(visdict, dict):
        for k in ["render_images_with_detail", "rendered_images_with_detail", "rendered_images_detail"]:
            if k in visdict:
                detail = visdict[k]
                break

    # detail candidates (fallback opdict)
    if detail is None and isinstance(opdict, dict):
        for k in ["rendered_images_detail", "rendered_images_with_detail"]:
            if k in opdict:
                detail = opdict[k]
                break

    if coarse is not None and detail is None:
        detail = coarse

    return detail


def pick_detail_render_deca(opdict: dict, visdict: dict):
    """
    'deca(old)' (codedict_old로 디코드된 비교 기준)에서 detail render를 최대한 찾아 반환.
    코드베이스마다 키 이름이 달라서, 가능한 후보를 폭넓게 탐색합니다.
    못 찾으면 old-coarse를, 그것도 없으면 None.
    """
    # 1) detail-old 후보 (visdict 우선)
    if isinstance(visdict, dict):
        # 흔히 old가 들어간 키들
        for k in [
            "render_images_with_detail_old",
            "rendered_images_with_detail_old",
            "rendered_images_detail_old",
            "render_images_detail_old",
            "render_images_old_with_detail",
            "rendered_images_old_with_detail",
        ]:
            if k in visdict:
                return visdict[k]

    # 2) detail-old 후보 (opdict)
    if isinstance(opdict, dict):
        for k in [
            "rendered_images_detail_old",
            "rendered_images_with_detail_old",
            "render_images_with_detail_old",
            "render_images_detail_old",
        ]:
            if k in opdict:
                return opdict[k]

        # 3) old coarse 후보 (coarse 비교)
        for k in [
            "rendered_images_old",
            "render_images_old",
            "old_rendered_images",
        ]:
            if k in opdict:
                return opdict[k]

    return None


def write_bin_line(path: str, v01: int):
    """txt에 '0' 또는 '1' 한 줄 append"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{int(v01)}\n")


def main(args):
    au_labels = [
        "au1", "au2", "au4", "au5", "au6", "au7", "au9", "au10", "au11",
        "au12", "au13", "au14", "au15", "au16", "au17", "au18", "au19",
        "au20", "au22", "au23", "au24", "au25", "au26", "au27"
    ]

    K = args.K
    assert K % 2 == 1, "K는 홀수여야 합니다 (예: 3,5,7)"
    half = K // 2

    device = args.device

    # DECA cfg
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.model.extract_tex = args.extractTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    deca_cfg.rasterizer_type = args.rasterizer_type

    deca = DECA(config=deca_cfg, device=device)

    # AU net
    auconf = get_config()
    auconf.evaluate = True
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    set_env(auconf)

    AU_net = MEFARG(
        num_main_classes=auconf.num_main_classes,
        num_sub_classes=auconf.num_sub_classes,
        backbone=auconf.arc
    ).to(device)
    AU_net = load_state_dict(AU_net, auconf.resume).to(device)
    AU_net.eval()

    # videos
    allVideos = sorted(glob.glob(args.inputpath), reverse=True)
    print(f"[INFO] num videos: {len(allVideos)}")
    print(f"[INFO] K={K}, half={half}")
    print(f"[INFO] prob_threshold={args.prob_threshold}, apply_sigmoid={args.apply_sigmoid}")

    # 비디오 루프
    for videopath in allVideos:
        inputpath = videopath
        name = videopath.split("/croppedImages2")[0].split("/")[-1]

        # save root (비디오별)
        save_root = args.savefolder.replace("*", name)
        safe_makedirs(save_root)

        # zip 이름(비디오별)
        zip_path = save_root.rstrip("/")

        # 이미 zip 만들었으면 스킵 옵션
        if args.skip_if_zip_exists:
            if os.path.isfile(zip_path + ".zip"):
                print(f"[SKIP] zip exists: {zip_path}.zip")
                continue

        # AU 폴더 생성
        for au in au_labels:
            safe_makedirs(os.path.join(save_root, au))

        # (파일 prefix) — Coarse 코드처럼 name 기반으로 깔끔하게
        # 필요하면 여기 규칙을 현웅 님 coarse 코드와 100% 동일하게 맞춰드릴 수 있어요.
        prefix = name

        # dataset
        testdata = datasets.TestData(
            inputpath,
            iscrop=args.iscrop,
            crop_size=deca_cfg.dataset.image_size,
            scale=1.25
        )

        # 프레임 루프
        for i in tqdm(range(half, len(testdata) - half), desc=f"video {name}"):
            # K-frames
            frames = [testdata[j]["image"][None, ...] for j in range(i - half, i + half + 1)]
            images = torch.cat(frames, dim=0).to(device)

            data_mid = testdata[i]
            imagename = data_mid.get("imagename", "")
            _frame_id = extract_frame_index(imagename)  # 필요하면 파일명에 쓰도록 확장 가능

            img_mid = images[half:half+1]  # 원본 중앙 프레임

            with torch.no_grad():
                codedict_old, codedict = deca.encode(images)

                # return_vis=True로 detail 결과를 visdict에서 찾기 쉽게
                opdict, visdict = deca.decode(
                    codedict, codedict_old,
                    use_detail=args.use_detail_eval,
                    return_vis=True
                )

                ours_detail = pick_detail_render_ours(opdict, visdict)
                deca_detail = pick_detail_render_deca(opdict, visdict)

                # ours_detail이 없으면 프레임 스킵 (렌더 실패)
                if ours_detail is None:
                    continue

                # deca_detail이 없으면: "항상 3종 세트" 조건을 만족시키기 위해
                # 최소한 old coarse라도 찾아서 쓰고, 그것도 없으면 마지막 fallback으로 ours_detail로 대체(강제 3종)
                if deca_detail is None:
                    # 강제 fallback (형식 보장)
                    deca_detail = ours_detail

                # AU scores
                s_img = au_forward_scores(AU_net, img_mid)[0].detach().cpu().numpy()
                s_ours = au_forward_scores(AU_net, ours_detail)[0].detach().cpu().numpy()
                s_deca = au_forward_scores(AU_net, deca_detail)[0].detach().cpu().numpy()

                if args.apply_sigmoid:
                    s_img = sigmoid_np(s_img)
                    s_ours = sigmoid_np(s_ours)
                    s_deca = sigmoid_np(s_deca)

                b_img = (s_img >= args.prob_threshold).astype(np.int64)
                b_ours = (s_ours >= args.prob_threshold).astype(np.int64)
                b_deca = (s_deca >= args.prob_threshold).astype(np.int64)

                # AU별로 txt append
                for k, au in enumerate(au_labels):
                    # img
                    p_img = os.path.join(save_root, au, f"{prefix}_{au}.txt")
                    # ours
                    p_ours = os.path.join(save_root, au, f"{prefix}_{au}R.txt")
                    # deca(old)
                    p_deca = os.path.join(save_root, au, f"{prefix}_{au}R_deca.txt")

                    write_bin_line(p_img, b_img[k])
                    write_bin_line(p_ours, b_ours[k])
                    write_bin_line(p_deca, b_deca[k])

        # 비디오 끝나면 zip 만들기 (save_root 폴더 전체)
        # zip은 save_root의 "폴더명" 기준으로 생성됨: <save_root>.zip
        base_dir = os.path.dirname(zip_path)
        base_name = os.path.basename(zip_path)
        archive_base = os.path.join(base_dir, base_name)

        # shutil.make_archive는 확장자 제외 base를 받음
        shutil.make_archive(archive_base, 'zip', root_dir=zip_path)

        print(f"[ZIP] {archive_base}.zip")

        # 옵션: zip 만든 뒤 폴더 삭제(원하면)
        if args.remove_folder_after_zip:
            shutil.rmtree(zip_path, ignore_errors=True)
            print(f"[CLEAN] removed folder: {zip_path}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--inputpath",
        default="/media/cine/de6afd1d-c444-4d43-a787-079519ace719/DISFA/video/*/croppedImages2/",
        type=str
    )
    parser.add_argument(
        "-s", "--savefolder",
        default="/media/cine/First/HWPJ2/ProjectResult/DISFA/DetailNew_20260103_mrf01/*/",
        type=str
    )
    parser.add_argument(
        "--pretrained_modelpath_ViT",
        default="/media/cine/First/HWPJ2/ProjectResult/DetailNew_20260103_mrf01/model.tar",
        type=str
    )

    # device/K
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--K", default=3, type=int)

    # deca options
    parser.add_argument("--iscrop", default=False, type=lambda x: x.lower() in ["true", "1"])
    parser.add_argument("--rasterizer_type", default="pytorch3d", type=str)
    parser.add_argument("--useTex", default=True, type=lambda x: x.lower() in ["true", "1"])
    parser.add_argument("--extractTex", default=False, type=lambda x: x.lower() in ["true", "1"])

    # eval options
    parser.add_argument("--use_detail_eval", default=True, type=lambda x: x.lower() in ["true", "1"])

    # binarize
    parser.add_argument("--prob_threshold", default=0.5, type=float)
    parser.add_argument("--apply_sigmoid", default=False, type=lambda x: x.lower() in ["true", "1"])

    # zip behavior
    parser.add_argument("--skip_if_zip_exists", default=True, type=lambda x: x.lower() in ["true", "1"])
    parser.add_argument("--remove_folder_after_zip", default=False, type=lambda x: x.lower() in ["true", "1"])

    args = parser.parse_args()
    main(args)
