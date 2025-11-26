import torch  # 파이토치 텐서/연산
from torch import nn  # 신경망 모듈
from einops import rearrange, repeat  # 텐서 형태 변환 유틸
from einops.layers.torch import Rearrange  # PyTorch용 Rearrange 레이어

#pair helper doesnt need

# ------------------------------------------------------------
# FeedForward: Transformer 블록의 MLP(FFN) 부분
# ------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()  # nn.Module 초기화
        self.net = nn.Sequential(                  # 순차적으로 레이어 쌓기
            nn.LayerNorm(dim),                       # 입력 정규화 (학습 안정성)
            nn.Linear(dim, hidden_dim),              # 차원 확장
            nn.GELU(),                               # 비선형 활성화
            nn.Dropout(dropout),                     # 드롭아웃
            nn.Linear(hidden_dim, dim),              # 차원 복원
            nn.Dropout(dropout)                      # 드롭아웃
        )
    def forward(self, x):
        return self.net(x)                         # FFN 통과 결과 반환

# ------------------------------------------------------------
# SelfAttention: AU 시퀀스 내부 토큰끼리 자기-어텐션
# ------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()                           # nn.Module 초기화
        inner_dim = heads * dim_head                     # 멀티헤드 총 채널 수
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads                           # 헤드 수 저장
        self.scale = dim_head ** -0.5                # 점곱 어텐션 스케일링 인자
        self.norm = nn.LayerNorm(dim)                # 입력 정규화

        self.attend = nn.Softmax(dim=-1)  # 어텐션 가중치 산출용 소프트맥스
        self.dropout = nn.Dropout(dropout)  # 어텐션 드롭아웃

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)  # Q,K,V 합쳐서 한 번에 생성
        self.to_out = nn.Sequential(                   # 출력 투영 + 드롭아웃
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 투영 불필요 시 항등

    def forward(self, x):
        x = self.norm(x)  # 프리-노름
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 1.Embedding + 분할 Q/K/V로 3분할
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # (b,heads,n,d)로 재배열
        # print(qkv[0].shape,q.shape,k.shape, v.shape )  # (디버깅) 모양 확인
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 2. Self attention 계산. 어텐션 점수 행렬(QK^T / sqrt(d))

        attn = self.attend(dots)  # 3.1. 소프트맥스로 가중치 계산
        attn = self.dropout(attn)  # 드롭아웃 적용

        out = torch.matmul(attn, v)  # 3.2. 가중합으로 컨텍스트 계산
        out = rearrange(out, 'b h n d -> b n (h d)')  # 헤드 축 병합
        return self.to_out(out)  # 출력 투영 후 반환
# ------------------------------------------------------------
# CrossAttentionKV: 서로 다른 소스에서 K와 V를 분리 생성하는 크로스-어텐션
#   - Q: AU 시퀀스 토큰들
#   - K: mean(Global) 1토큰
#   - V: mean(Detail) 1토큰
# ------------------------------------------------------------
class CrossAttentionKV(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()                           # nn.Module 초기화
        inner_dim = heads * dim_head                     # 멀티헤드 총 채널 수
        project_out = not (heads == 1 and dim_head == dim)  # 투영 필요 여부
        self.heads = heads                           # 헤드 수
        self.scale = dim_head ** -0.5                # 스케일링 인자

        self.norm_q = nn.LayerNorm(dim)              # Q 정규화
        self.norm_k = nn.LayerNorm(dim)              # K 정규화
        self.norm_v = nn.LayerNorm(dim)              # V 정규화

        self.attend = nn.Softmax(dim=-1)               # 어텐션 가중치
        self.dropout = nn.Dropout(dropout)  # 드롭아웃

        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Q 생성 투영
        self.to_k = nn.Linear(dim, inner_dim, bias=False)  # K 생성 투영
        self.to_v = nn.Linear(dim, inner_dim, bias=False)  # V 생성 투영

        self.to_out = nn.Sequential(                   # 출력 투영 + 드롭아웃
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 필요 없으면 항등

    def forward(self, x, y_k, y_v):
        q = self.to_q(self.norm_q(x))                # 입력 x(AU 시퀀스)에서 Q 생성
        k = self.to_k(self.norm_k(y_k))              # y_k(글로벌 mean)에서 K 생성
        v = self.to_v(self.norm_v(y_v))              # y_v(디테일 mean)에서 V 생성

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)  # [B,H,N,Dh]
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)  # [B,H,1,Dh]
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)  # [B,H,1,Dh]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 교차 어텐션 점수 계산
        attn = self.attend(dots)  # 어텐션 가중치
        attn = self.dropout(attn)  # 드롭아웃

        out = torch.matmul(attn, v)  # 가중합 컨텍스트
        out = rearrange(out, 'b h n d -> b n (h d)')  # 헤드 병합
        return self.to_out(out)                        # 투영 후 반환(잔차는 상위에서 더함)

# ------------------------------------------------------------
# TransformerDetail: (SelfAttn → CrossAttnKV → FFN) 블록을 depth만큼 스택
# ------------------------------------------------------------
class TransformerDetail(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()                           # nn.Module 초기화
        self.norm = nn.LayerNorm(dim)                # 최종 정규화

        self.layers = nn.ModuleList([])  # 레이어 리스트 초기화
        for _ in range(depth):  # 지정된 깊이만큼 블록을 쌓음
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # 자기어텐션
                CrossAttentionKV(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # 교차어텐션
                FeedForward(dim, mlp_dim, dropout=dropout)  # FFN
            ]))

    def forward(self, x, y_k, y_v):
        # x   : AU 시퀀스 토큰 + (이후 앞단에서 CLS 추가됨)
        # y_k : 글로벌 mean  토큰 (shape=[B,1,dim])
        # y_v : 디테일 mean  토큰 (shape=[B,1,dim])
        for attn, cattn, ff in self.layers:             # 각 블록 반복
            x = attn(x) + x                             # Self-Attn + residual
            x = cattn(x, y_k, y_v) + x                  # Cross-Attn(K=glob mean, V=det mean) + residual
            x = ff(x) + x                               # FFN + residual
        return self.norm(x)                             # 마지막 정규화

# ------------------------------------------------------------
# ViT_Detail: 코어스 ViT와 동일한 자리에 mean을 계산해 넣는 디테일 전용 ViT
#   - Query  : AU 시퀀스 (frames=T)
#   - Key    : mean(Global)   (1토큰)
#   - Value  : mean(Detail)   (1토큰)
#   - Output : 중앙 프레임용 detailcode (num_features)
# ------------------------------------------------------------
class ViT_Detail(nn.Module):
    def __init__(self, *, num_features, frames, dim, depth, heads,
                 mlp_dim, dropout, emb_dropout, dim_head):
        super().__init__()                           # nn.Module 초기화
        self.frames = frames                         # 프레임 수(T) (wt dependant)
        self.dim = dim                               # 토큰 임베딩 차원

        mid_channel = 512   # afn's width

        # AU feature image -> patch & embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 27, p2 = 512),
            nn.LayerNorm(13824),
            nn.Linear(13824, mid_channel),
            nn.LayerNorm(mid_channel),
        )

        # 글로벌/디테일 피처를 ViT 차원으로 사상 (E_flame/E_detail 백본 피처 → dim)
        self.FC_glob = nn.Linear(2048, dim, bias=True)  # 글로벌 백본피처 2048 → dim
        self.FC_det  = nn.Linear(2048, dim, bias=True)  # 디테일 백본피처 2048 → dim

        # 포지셔널 임베딩(프레임 + CLS 1개)
        self.pos_embedding = nn.Parameter(torch.randn(1, frames + 1, dim))  # [1, T+1, dim]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))               # CLS 학습 토큰
        self.dropout = nn.Dropout(emb_dropout)                               # 임베딩 드롭아웃

        # 트랜스포머 스택 (Self → Cross(K/V 분리) → FFN)
        self.transformer = TransformerDetail(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 출력 헤드 (CLS에서 detailcode 회귀)
        self.to_latent = nn.Identity()               # 필요 시 추가 변환용 자리(현재 통과)
        self.mlp_head = nn.Linear(dim, num_features)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight) # initalize
        """
        #GPT suggest new mlp head.
        # MLP Head (LayerNorm → Linear(dim→mlp_dim) → GELU → Linear(mlp_dim→num_features))
        # 목적: 트랜스포머가 만든 CLS 표현을 비선형 변환으로 강화해, 복잡한 회귀/분류(예: detail code 256차) 표현력을 높임
        # 장점: 단일 Linear 대비 비선형 매핑 가능 → 성능 잠재력↑ (특히 고차·비선형 관계)
        # 단점: 파라미터 증가로 과적합 위험↑, 초기 출력 스케일 흔들림 가능
        # 팁:
        #  - 정규화: Head 앞 LayerNorm으로 입력 분포 안정화 (Pre-LN 블록과 과도한 중복은 지양)
        #  - 규제: 중간에 Dropout(예: 0.1) 또는 Weight Decay로 과적합 억제
        #  - 초기화: 마지막 Linear를 약하게(init std↓, bias=0) 설정하면 학습 초반 안정적
        #    ex) nn.init.xavier_uniform_(head[-1].weight, gain=0.5); nn.init.zeros_(head[-1].bias)
        #  - 출력 범위가 정해진 회귀면 마지막에 tanh/sigmoid 추가 또는 라벨 스케일링으로 정합
        #  - 비교 기준: 단일 Linear head는 경량·안정, MLP head는 표현력↑ — 데이터/과제 난이도에 따라 A/B로 선택
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),                       # 최종 정규화
            nn.Linear(dim, mlp_dim), nn.GELU(),      # 확장 + 비선형
            nn.Linear(mlp_dim, num_features)             # detailcode 회귀(num_features=256)
        )
        """

    def forward(self, au_seq, global_feature_seq, detail_feature_seq):
        """
        au_seq            : (T, 27, 512)    # AU 시퀀스 피처 (현재 코어스 입력과 동일 가정)
        global_feature_seq: (T, 2048)   # E_flame 백본 피처 시퀀스
        detail_feature_seq: (T, 2048)   # E_detail 백본 피처 시퀀스
        """
        T = self.frames
        #1. make au feaure sequence -> patch embedding
        x = self.to_patch_embedding(au_seq[:,None,...]).view(1,T,-1)  # 채널 축 추가 후 패치 임베딩 → (배치=1, 토큰=7, 임베딩)
        #2. fusion layer: mean the global & detail feature (from E_flame, E_detail)
        y_k = torch.mean(self.FC_glob(global_feature_seq), dim=0).view(1,1,-1)
        y_v = torch.mean(self.FC_det(detail_feature_seq), dim=0).view(1,1,-1)
        #3. duplicate cls token to match the batch size
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1  d', b = 1)  # 배치 크기에 맞춰 CLS 토큰 복제
            # print(x.shape)  # (디버깅) x 모양 확인
        #4. concat the cls token infrontof AU patch tokens
        x = torch.cat((cls_tokens, x), dim=1)  # CLS 토큰을 시퀀스 앞에 붙임 → 길이 8
        #5. add positional embedding
        x = x + self.pos_embedding[:, :T+1]  # 위치 임베딩 더하기
        #6. embedding dropout
        x = self.dropout(x)  # 드롭아웃 적용
        #7. get transformer result
        x = self.transformer(x, y_k, y_v)  # 트랜스포머에 x(타깃), y(글로벌 컨텍스트 요약) 전달
        #8.pool the cls token position expression
        x = x[:, 0]  # CLS 토큰 위치의 표현만 추출(풀링)
        x = self.to_latent(x)  # (항등) 잠재 표현 유지
        return self.mlp_head(x) # 최종 헤드 통과 로짓 반환