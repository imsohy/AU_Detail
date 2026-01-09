"""
원본에서 AU 뽑고 (Sudo GT)
우리 모델 Detail 포함 렌더링 해서 AU 뽑고 결과값 .csv 로 저장.
"""
# -*- coding: utf-8 -*-
import glob
import os, sys
import re
import csv
import argparse
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
    """
    파일명에서 마지막 숫자 덩어리를 frame index로 사용.
    DISFA 네이밍이 다르면 이 함수만 현웅 님 규칙에 맞게 바꾸면 됩니다.
    """
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


def pick_detail_render(opdict: dict, visdict: dict):
    coarse = None
    detail = None

    # --- Coarse 후보 ---
    if isinstance(opdict, dict):
        coarse = opdict.get("rendered_images", None)
        if coarse is None:
            coarse = opdict.get("render_images", None)

    # --- Detail 후보 (우선 visdict) ---
    if isinstance(visdict, dict):
        detail = visdict.get("render_images_with_detail", None)
        if detail is None:
            detail = visdict.get("rendered_images_with_detail", None)
        if detail is None:
            detail = visdict.get("rendered_images_detail", None)

    # --- Detail 후보 (fallback: opdict) ---
    if detail is None and isinstance(opdict, dict):
        detail = opdict.get("rendered_images_detail", None)
        if detail is None:
            detail = opdict.get("rendered_images_with_detail", None)

    # --- 마지막 fallback: detail 못 찾으면 coarse 사용 ---
    if coarse is not None and detail is None:
        detail = coarse

    return coarse, detail



def main(args):
    # (작동 코드의 AU 리스트 유지)
    au_labels = [
        "au1", "au2", "au4", "au5", "au6", "au7", "au9", "au10", "au11",
        "au12", "au13", "au14", "au15", "au16", "au17", "au18", "au19",
        "au20", "au22", "au23", "au24", "au25", "au26", "au27"
    ]
    num_au = len(au_labels)

    # 시퀀스 길이 K (작동 코드와 동일한 방식)
    K = args.K
    assert K % 2 == 1, "K는 홀수여야 합니다 (예: 3,5,7)"
    half = K // 2

    device = args.device

    # DECA cfg 세팅(작동 코드 스타일)
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.model.extract_tex = args.extractTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    deca_cfg.rasterizer_type = args.rasterizer_type

    deca = DECA(config=deca_cfg, device=device)

    # AU net 세팅(작동 코드 스타일)
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

    # 입력 비디오들
    allVideos = sorted(glob.glob(args.inputpath), reverse=True)

    print(f"[INFO] num videos: {len(allVideos)}")
    print(f"[INFO] K={K}, half={half}")
    print(f"[INFO] use_detail_eval={args.use_detail_eval}, return_vis={args.return_vis}")
    print(f"[INFO] save CSV: {args.save_csv}")

    # 저장 폴더(작동 코드처럼 * 치환 가능)
    safe_makedirs(args.savefolder.replace("*", "TMP"))

    # 전체 CSV 열 구성
    # - frame, imagename
    # - (원본) score/bin
    # - (detail렌더) score/bin
    header = ["video_name", "frame", "imagename"]
    header += [f"img_score_{au}" for au in au_labels]
    header += [f"det_score_{au}" for au in au_labels]
    if args.save_bin:
        header += [f"img_bin_{au}" for au in au_labels]
        header += [f"det_bin_{au}" for au in au_labels]

    # CSV open
    safe_makedirs(os.path.dirname(args.save_csv) if os.path.dirname(args.save_csv) else ".")
    with open(args.save_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

        # 비디오 루프
        for videopath in allVideos:
            inputpath = videopath
            name = videopath.split("/croppedImages2")[0].split("/")[-1]  # 작동 코드 로직 유지

            # 데이터 로더
            testdata = datasets.TestData(
                inputpath,
                iscrop=args.iscrop,
                crop_size=deca_cfg.dataset.image_size,
                scale=1.25
            )

            # 프레임 루프
            for i in tqdm(range(half, len(testdata) - half), desc=f"video {name}"):
                # 시퀀스 구성 (K프레임)
                frames = [testdata[j]["image"][None, ...] for j in range(i - half, i + half + 1)]
                images = torch.cat(frames, dim=0).to(device)

                data_mid = testdata[i]
                imagename = data_mid.get("imagename", "")
                frame_id = extract_frame_index(imagename)

                # 원본은 중앙    프레임만
                img_mid = images[half:half+1]

                with torch.no_grad():
                    codedict_old, codedict = deca.encode(images)

                    if args.return_vis:
                        opdict, visdict = deca.decode(
                            codedict, codedict_old,
                            use_detail=args.use_detail_eval,
                            return_vis=True
                        )
                    else:
                        opdict = deca.decode(
                            codedict, codedict_old,
                            use_detail=args.use_detail_eval
                        )
                        visdict = {}

                    coarse_img, detail_img = pick_detail_render(opdict, visdict)

                    # detail_img가 None이면 스킵 (렌더 자체가 안 나온 상황)
                    if detail_img is None:
                        continue

                    # AU score 추출: 원본 vs detail렌더
                    score_img_t = au_forward_scores(AU_net, img_mid)      # [1, num_au]
                    score_det_t = au_forward_scores(AU_net, detail_img)   # [1, num_au]

                    score_img = score_img_t[0].detach().cpu().numpy()
                    score_det = score_det_t[0].detach().cpu().numpy()

                    # 출력이 logit일 가능성이 있으면 sigmoid 적용 옵션
                    if args.apply_sigmoid:
                        score_img = sigmoid_np(score_img)
                        score_det = sigmoid_np(score_det)

                    # 0 1 구분되기 이전의 값 기록하기
                    row = [name, frame_id, imagename]
                    row += score_img.tolist()
                    row += score_det.tolist()

                    # bin도 저장(원하면)
                    if args.save_bin:
                        img_bin = (score_img >= args.prob_threshold).astype(np.int64)
                        det_bin = (score_det >= args.prob_threshold).astype(np.int64)
                        row += img_bin.tolist()
                        row += det_bin.tolist()

                    writer.writerow(row)

    print("\n=== DONE ===")
    print(f"[CSV] {args.save_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 작동 코드의 기본 경로 스타일을 그대로 참고
    parser.add_argument(
        "-i", "--inputpath",
        default="/media/cine/de6afd1d-c444-4d43-a787-079519ace719/DISFA/video/*/croppedImages2/",
        type=str
    )
    parser.add_argument(
        "-s", "--savefolder",
        default="/media/cine/First/HWPJ2/ProjectResult/DISFA/*/",
        type=str
    )
    parser.add_argument(
        "--pretrained_modelpath_ViT",
        default="/media/cine/First/HWPJ2/ProjectResult/DISFA/DetailNew/model.tar",
        type=str
    )

    # 출력 CSV (한 파일로 깔끔하게)
    parser.add_argument(
        "--save_csv",
        default="/media/cine/First/HWPJ2/ProjectResult/DISFA/ALL_DISFA_AU_scores.csv",
        type=str
    )

    # 장치/시퀀스
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--K", default=3, type=int)

    # 데카 옵션
    parser.add_argument("--iscrop", default=False, type=lambda x: x.lower() in ["true", "1"])
    parser.add_argument("--rasterizer_type", default="pytorch3d", type=str)
    parser.add_argument("--useTex", default=True, type=lambda x: x.lower() in ["true", "1"])
    parser.add_argument("--extractTex", default=False, type=lambda x: x.lower() in ["true", "1"])

    # 평가 옵션
    parser.add_argument("--use_detail_eval", default=True, type=lambda x: x.lower() in ["true", "1"])
    parser.add_argument("--return_vis", default=True, type=lambda x: x.lower() in ["true", "1"])

    # score/bin 저장 옵션
    parser.add_argument("--save_bin", default=True, type=lambda x: x.lower() in ["true", "1"])
    parser.add_argument("--prob_threshold", default=0.5, type=float)

    # AU 출력이 logit일 때 sigmoid 적용
    parser.add_argument("--apply_sigmoid", default=False, type=lambda x: x.lower() in ["true", "1"])

    args = parser.parse_args()
    main(args)

