#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_au_two_folders.py  (with AU name mapping)

원본 이미지 폴더(src)와 생성/렌더 결과 폴더(dst)의 파일을 1:1로 매칭한 뒤,
ME-GraphAU(MEFARG)로 각각 AU 확률을 추론하여, 이진화 후 (src vs dst) 일치도를
AU별 Precision/Recall/F1로 집계합니다. 결과는 CSV로 저장됩니다.

추가: AU_index -> AU_name 매핑 컬럼을 CSV에 포함 (DISFA12/BP4D12 프리셋 또는 사용자 지정)

예시:
python compare_au_two_folders.py \
  --src_dir /path/to/originals \
  --dst_dir /path/to/results \
  --resume /path/to/OpenGraphAU-ResNet18_second_stage.pth \
  --output_csv /path/to/out/au_compare.csv \
  --num_classes 12 --arc resnet18 --device cuda:0 \
  --au_set DISFA12
"""

import os, sys, csv, argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# === 프로젝트 유틸 불러오기 (전처리/지표 재사용) ===
try:
    from utils import image_test, load_state_dict
except Exception as e:
    print("ERROR: utils.py를 import하지 못했습니다. PYTHONPATH를 확인하세요.", e)
    raise

# === MEFARG (ME-GraphAU) ===
try:
    from model.MEFL import MEFARG
except Exception as e:
    print("ERROR: 'from model.MEFL import MEFARG' 실패. 프로젝트 구조(model/MEFL.py)를 확인하세요.", e)
    raise


# -------------------------
# AU name presets
# -------------------------
DISFA12 = ["AU1","AU2","AU4","AU5","AU6","AU9","AU12","AU15","AU17","AU20","AU25","AU26"]
BP4D12  = ["AU1","AU2","AU4","AU6","AU7","AU10","AU12","AU14","AU15","AU17","AU23","AU24"]

def resolve_au_names(num_classes, au_set, au_names):
    if au_set == "CUSTOM":
        if not au_names or len(au_names) != num_classes:
            raise ValueError(f"--au_set CUSTOM 사용 시 --au_names를 num_classes({num_classes})개로 제공해야 합니다.")
        return list(au_names)
    if au_set == "DISFA12":
        names = DISFA12
    elif au_set == "BP4D12":
        names = BP4D12
    else:
        # Fallback: generic AU_0..AU_{C-1}
        names = [f"AU_{i}" for i in range(num_classes)]

    if len(names) != num_classes:
        raise ValueError(f"{au_set} 프리셋의 길이({len(names)})가 num_classes({num_classes})와 다릅니다.")
    return names


def list_images(root, exts, recursive=False):
    root = Path(root)
    if recursive:
        paths = [p for ext in exts for p in root.rglob(f"*.{ext}")]
    else:
        paths = [p for ext in exts for p in root.glob(f"*.{ext}")]
    return sorted(paths)


def build_pairs(src_dir, dst_dir, exts, recursive=False, match_by="stem"):
    keyfun = (lambda p: p.stem.lower()) if match_by == "stem" else (lambda p: p.name.lower())
    srcs = list_images(src_dir, exts, recursive)
    dsts = list_images(dst_dir, exts, recursive)
    dst_map = {}
    for p in dsts:
        k = keyfun(p)
        if k not in dst_map: dst_map[k] = p

    pairs, missing = [], []
    for s in srcs:
        k = keyfun(s)
        d = dst_map.get(k)
        (pairs if d is not None else missing).append((s, d) if d is not None else s)
    return pairs, missing


@torch.no_grad()
def infer_probs(model, pil_img, preprocess, device):
    x = preprocess(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]
    logits = model(x)                                # [1,C]
    return torch.sigmoid(logits).squeeze(0)          # [C]


def update_confmat(stats, y_true, y_pred):
    for j in range(y_true.numel()):
        yt = int(y_true[j].item()); yp = int(y_pred[j].item())
        d = stats[j]
        if yp == 1 and yt == 1: d['TP'] += 1
        elif yp == 1 and yt == 0: d['FP'] += 1
        elif yp == 0 and yt == 1: d['FN'] += 1
        else: d['TN'] += 1


def prf_from_counts(TP, FP, FN):
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 모델/전처리
    net = MEFARG(num_classes=args.num_classes, backbone=args.arc)
    if args.resume and Path(args.resume).exists():
        net = load_state_dict(net, args.resume)
    else:
        print("WARNING: 체크포인트(--resume)를 찾지 못했습니다. 랜덤 가중치로 동작합니다.")
    net.eval().to(device)
    preprocess = image_test(img_size=args.img_size, crop_size=args.crop_size)

    # AU 이름 매핑
    au_names = resolve_au_names(args.num_classes, args.au_set, args.au_names)

    # 페어링
    pairs, missing = build_pairs(args.src_dir, args.dst_dir, args.exts, args.recursive, args.match_by)
    if len(pairs) == 0:
        print("매칭된 페어가 없습니다. --exts / --match_by / --recursive 옵션을 확인하세요.")
        return 1

    # 임계값
    thresh_vec = (torch.tensor(args.thresh, dtype=torch.float32)
                  if (args.thresh and len(args.thresh)>0)
                  else torch.full((args.num_classes,), args.au_thresh, dtype=torch.float32))

    # 통계
    stats = {j: {'TP':0, 'FP':0, 'FN':0, 'TN':0} for j in range(args.num_classes)}

    # 루프
    for s, d in tqdm(pairs, desc="Comparing AU (src vs dst)"):
        try:
            img_s = Image.open(s).convert("RGB")
            img_d = Image.open(d).convert("RGB")
        except Exception as e:
            print(f"[WARN] 이미지 열기 실패: {s} | {d} ({e})")
            continue

        ps = infer_probs(net, img_s, preprocess, device)
        pd = infer_probs(net, img_d, preprocess, device)

        ys = (ps >= thresh_vec.to(ps.device)).long()
        yd = (pd >= thresh_vec.to(pd.device)).long()

        update_confmat(stats, ys, yd)

    # CSV 저장
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["AU_index","AU_name","TP","FP","FN","TN","Precision","Recall","F1"])
        macro_prec = macro_rec = macro_f1 = 0.0

        for j in range(args.num_classes):
            TP = stats[j]['TP']; FP = stats[j]['FP']; FN = stats[j]['FN']; TN = stats[j]['TN']
            prec, rec, f1 = prf_from_counts(TP, FP, FN)
            macro_prec += prec; macro_rec += rec; macro_f1 += f1
            w.writerow([j, au_names[j], TP, FP, FN, TN, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

        w.writerow([])
        w.writerow(["macro_precision", f"{macro_prec/args.num_classes:.4f}"])
        w.writerow(["macro_recall",    f"{macro_rec/args.num_classes:.4f}"])
        w.writerow(["macro_f1",        f"{macro_f1/args.num_classes:.4f}"])
        w.writerow(["au_set", args.au_set])

    print(f"[DONE] Saved: {args.output_csv}")
    if len(missing) > 0:
        print(f"[NOTE] dst에서 못 찾은 src 파일 {len(missing)}개 (예: {missing[0] if isinstance(missing[0], Path) else missing[0]})")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Compare AU between two image folders using ME-GraphAU (MEFARG)")
    p.add_argument("--src_dir", required=True, help="원본(입력) 이미지 폴더")
    p.add_argument("--dst_dir", required=True, help="생성/렌더 결과 이미지 폴더")
    p.add_argument("--resume", required=True, help="MEFARG 체크포인트(.pth)")
    p.add_argument("--output_csv", required=True, help="결과 CSV 경로")
    p.add_argument("--device", default="cuda:0", help="cuda:N 또는 cpu")
    p.add_argument("--exts", nargs="+", default=["jpg","jpeg","png","bmp","tif","tiff"], help="허용 확장자 목록")
    p.add_argument("--recursive", action="store_true", help="하위 폴더까지 재귀 탐색")
    p.add_argument("--match_by", choices=["stem","name"], default="stem", help="파일 매칭 기준: stem(확장자 제외) 또는 name(확장자 포함)")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--num_classes", type=int, default=12, help="AU 클래스 수 (체크포인트에 맞춰 설정)")
    p.add_argument("--arc", default="resnet18", help="백본 종류 (체크포인트에 맞춰 설정, 예: resnet18/resnet50/swin_t 등)")
    p.add_argument("--au_thresh", type=float, default=0.5, help="기본 이진화 임계값")
    p.add_argument("--thresh", type=float, nargs="*", help="AU별 임계값을 C개 나열 (예: --thresh 0.5 0.4 ...)")
    # NEW: AU name mapping
    p.add_argument("--au_set", choices=["DISFA12","BP4D12","CUSTOM"], default="DISFA12",
                   help="AU 이름 프리셋(또는 CUSTOM으로 직접 지정)")
    p.add_argument("--au_names", nargs="*", help="--au_set CUSTOM일 때 사용할 AU 이름 목록 (num_classes개)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(args))
