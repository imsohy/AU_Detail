import pandas as pd
import numpy as np
import os

# =========================
# CONFIG
# =========================
csv_path = "/media/cine/First/HWPJ2/ProjectResult/DISFA/ALL_DISFA_AU_scores.csv"
# Colab: csv_path = "/content/ALL_DISFA_AU_scores.csv"

au_labels = [
    "au1","au2","au4","au5","au6","au7","au9","au10","au11",
    "au12","au13","au14","au15","au16","au17","au18","au19",
    "au20","au22","au23","au24","au25","au26","au27"
]

def f1(tp, fp, fn, eps=1e-8):
    return (2*tp) / (2*tp + fp + fn + eps)

# =========================
# LOAD + FIX (MultiIndex / duplicated columns)
# =========================
df = pd.read_csv(csv_path)

# 1) MultiIndex 형태로 읽힌 경우: reset_index()로 진짜 키를 컬럼으로 꺼냄
#    출력 예시: index가 (video_name, frame, imagename, ...)
if isinstance(df.index, pd.MultiIndex):
    df = df.reset_index()

    # 보통 level_0/1/2가 video/frame/imagename일 확률이 높음
    # (출력: LeftVideoSNxxx / 3 / LeftVideoSNxxx_frame0003)
    rename_map = {}
    if "level_0" in df.columns: rename_map["level_0"] = "video_name"
    if "level_1" in df.columns: rename_map["level_1"] = "frame"
    if "level_2" in df.columns: rename_map["level_2"] = "imagename"
    if rename_map:
        df = df.rename(columns=rename_map)

# 2) "video_name" 같은 컬럼명이 중복으로 존재하는 문제 해결:
#    첫 번째(문자열 video_name)를 남기고 나머지 중복(실수값 video_name 등)을 제거
df = df.loc[:, ~df.columns.duplicated()]

# 3) video_name이 이상하면(문자열이 아니라 float로 들어온 경우),
#    인덱스에서 꺼낸 컬럼 후보를 찾아 교체 시도 (안전장치)
if "video_name" not in df.columns or not pd.api.types.is_object_dtype(df["video_name"]):
    # reset_index 후 생긴 컬럼 중 "LeftVideo" 같은 문자열이 많은 컬럼을 찾아서 video_name로 지정
    candidate_cols = []
    for c in df.columns[:20]:  # 앞쪽에서 주로 나오므로 앞 20개 정도만 탐색
        if pd.api.types.is_object_dtype(df[c]):
            # 문자열 형태 비율이 높은 컬럼 후보
            sample = df[c].dropna().astype(str).head(50)
            if (sample.str.contains("Video", case=False, na=False).mean() > 0.3) or \
               (sample.str.contains("LeftVideo", case=False, na=False).mean() > 0.1):
                candidate_cols.append(c)
    if candidate_cols:
        df = df.rename(columns={candidate_cols[0]: "video_name"})

# =========================
# VALIDATION
# =========================
required_cols = ["video_name"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"[ERROR] Missing required columns: {missing}\n"
                   f"Columns head: {list(df.columns[:30])}")

# AU bin 컬럼 존재 체크
need_cols = []
for au in au_labels:
    need_cols += [f"img_bin_{au}", f"det_bin_{au}"]
missing_bins = [c for c in need_cols if c not in df.columns]
if missing_bins:
    raise KeyError(
        "[ERROR] Missing AU bin columns (first 10 shown): "
        f"{missing_bins[:10]}\n"
        "=> CSV에 img_bin_auXX / det_bin_auXX 컬럼이 있는지 확인하세요."
    )

# =========================
# F1 COMPUTE
# =========================
rows = []
for video, g in df.groupby("video_name"):
    for au in au_labels:
        y = g[f"img_bin_{au}"].to_numpy().astype(int)   # pseudo GT
        p = g[f"det_bin_{au}"].to_numpy().astype(int)   # detail pred

        tp = int(np.sum((p == 1) & (y == 1)))
        fp = int(np.sum((p == 1) & (y == 0)))
        fn = int(np.sum((p == 0) & (y == 1)))

        rows.append([video, au, float(f1(tp, fp, fn)), tp, fp, fn])

out = pd.DataFrame(rows, columns=["video_name", "AU", "F1_agreement", "TP", "FP", "FN"])

# =========================
# SAVE
# =========================
out_path = os.path.join(os.path.dirname(csv_path), "video_au_f1_agreement.csv")
out.to_csv(out_path, index=False)

print(f"[OK] saved: {out_path}")

# =========================
# OPTIONAL SUMMARY PRINTS
# =========================
# 전체 요약
print("\n[SUMMARY] F1_agreement stats")
print(out["F1_agreement"].describe())

# AU별 평균
au_mean = out.groupby("AU")["F1_agreement"].mean().sort_values(ascending=False)
print("\n[SUMMARY] AU-wise mean F1 (top 10)")
print(au_mean.head(10))

# 비디오별 평균
video_mean = out.groupby("video_name")["F1_agreement"].mean().sort_values(ascending=False)
print("\n[SUMMARY] Video-wise mean F1 (top 10)")
print(video_mean.head(10))
