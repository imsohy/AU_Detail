import torch.nn as nn  # 신경망 모듈
from .ViTVideoSequence_Cross_WT_DetailNew import ViT_Detail  # 위에서 정의한 ViT_Detail 불러오기

# ------------------------------------------------------------
# ViTDetailEncoderSeque: 코어스용 래퍼와 동일한 패턴의 래퍼
#   - 호출 인터페이스만 간단히 맞춰주어 상위 코드 변경 최소화
# ------------------------------------------------------------
class ViTDetailEncoderSeque(nn.Module):
    def __init__(self, num_features=256, frames=3, dim=512, depth=6, heads=4,
                 mlp_dim=1024, dropout=0.1, emb_dropout=0.1, dim_head=None):
        super().__init__()  # nn.Module 초기화
        if dim_head == None:
            dim_head = dim // heads
        # 내부에 실제 Detail ViT를 구성
        self.vit = ViT_Detail(
            num_features=num_features,       # detailcode (final output dimension)
            frames=frames,           # 윈도우 프레임 수(코어스와 동일)
            dim=dim,                 # 토큰 임베딩 차원
            depth=depth,             # Transformer 층 수
            heads=heads,             # 멀티헤드 수
            mlp_dim=mlp_dim,         # FFN 내부 차원
            dropout=dropout,         # 드롭아웃(블록 내부)
            emb_dropout=emb_dropout,  # 드롭아웃(임베딩)
            dim_head = dim_head
        )

    def forward(self, au_seq, global_feature_seq, detail_feature_seq):
        # 상위 모듈에서 (B,T,·) 형태의 시퀀스를 그대로 전달받아 ViT에 입력
        return self.vit(au_seq, global_feature_seq, detail_feature_seq)