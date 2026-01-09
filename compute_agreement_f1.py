"""
.csv ->f1 스코어 반영
"""
import pandas as pd
import numpy as np
import os

csv_path = "/media/cine/First/HWPJ2/PAPER_DISFA/ALL_DISFA_AU_scores.csv"
df = pd.read_csv(csv_path)

au_labels = [
    "au1","au2","au4","au5","au6","au7","au9","au10","au11",
    "au12","au13","au14","au15","au16","au17","au18","au19",
    "au20","au22","au23","au24","au25","au26","au27"
]

def f1(tp, fp, fn, eps=1e-8):
    return (2*tp) / (2*tp + fp + fn + eps)

rows = []
for video, g in df.groupby("video_name"):
    for au in au_labels:
        y = g[f"img_bin_{au}"].to_numpy().astype(int)   # pseudo GT
        p = g[f"det_bin_{au}"].to_numpy().astype(int)   # detail pred

        tp = np.sum((p==1) & (y==1))
        fp = np.sum((p==1) & (y==0))
        fn = np.sum((p==0) & (y==1))

        rows.append([video, au, f1(tp,fp,fn), tp, fp, fn])

out = pd.DataFrame(rows, columns=["video_name","AU","F1_agreement","TP","FP","FN"])

# 🔽 절대경로로 저장 (원본 CSV와 같은 디렉토리)
out_path = os.path.join(
    os.path.dirname(csv_path),
    "video_au_f1_agreement.csv"
)
out.to_csv(out_path, index=False)

print(f"saved: {out_path}")

