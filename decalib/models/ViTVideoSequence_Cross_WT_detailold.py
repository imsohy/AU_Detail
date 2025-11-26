import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1) # torch.Size([1, 4, 768])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # print(qkv[0].shape,q.shape,k.shape, v.shape ) # torch.Size([1, 4, 256]) torch.Size([1, 4, 4, 64]) torch.Size([1, 4, 4, 64]) torch.Size([1, 4, 4, 64])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        x = self.norm(x)
        y = self.norm(y)
        # print(x.shape, y.shape)
        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim = -1)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        # print(q.shape)
        # print(kv)
        # print(kv.shape)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # print(qkv[0].shape,q.shape,k.shape, v.shape ) # torch.Size([1, 4, 256]) torch.Size([1, 4, 4, 64]) torch.Size([1, 4, 4, 64]) torch.Size([1, 4, 4, 64])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Cross_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x,y):
        for attn, cattn, ff in self.layers:
            x = attn(x) + x
            x = cattn(x,y) + x
            x = ff(x) + x

        return self.norm(x)

# new model!!!
class ViT_Multi(nn.Module):
    def __init__(self, num_classes_g, num_classes_d, frames=3,
                 dim=512, depth=6, heads=4, dim_head=64,
                 mlp_dim=1024, dropout=0.1, emb_dropout=0.1,
                 au_in_dim=13824, det_in_dim=None, glob_in_dim=2048):
        super().__init__()
        # 1) 투영기
        self.FC_glob = nn.Linear(glob_in_dim, dim)             # global → D
        self.proj_au   = nn.Sequential(nn.LayerNorm(au_in_dim),
                                       nn.Linear(au_in_dim, dim),
                                       nn.LayerNorm(dim))
        # detail은 E_detail 결과(n_detail)를 바로 linear
        det_in_dim = det_in_dim or  self._infer_det_in_dim()   # 필요 시 cfg에서 주입
        self.proj_det  = nn.Sequential(nn.LayerNorm(det_in_dim),
                                       nn.Linear(det_in_dim, dim),
                                       nn.LayerNorm(dim))
        # 2) 토큰/포지션/타입
        self.cls_g = nn.Parameter(torch.randn(1,1,dim))
        self.cls_d = nn.Parameter(torch.randn(1,1,dim))
        max_len = 2 + 2*frames
        self.pos_embedding  = nn.Parameter(torch.randn(1, max_len, dim))
        # 타입 임베딩( AU vs DETAIL 구분 신호 )
        self.type_embedding = nn.Parameter(torch.randn(1, 2*frames, dim))  # [AU×T | DET×T]
        self.dropout = nn.Dropout(emb_dropout)
        # 3) 트랜스포머
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # 4) 헤드 2개
        self.head_g = nn.Linear(dim, num_classes_g)  # coarse 파라미터
        self.head_d = nn.Linear(dim, num_classes_d)  # detail latent

    def forward(self, au_seq, det_seq, global_feature):
        """
        au_seq:   (T, au_in_dim)      # 기존 afn을 평탄화/패치 임베딩 끝난 형상(아래 래퍼에서 맞춤)
        det_seq:  (T, det_in_dim)     # E_detail per-frame
        global_feature: (T, glob_in_dim)
        """

        B = 1 if au_seq.dim()==2 else au_seq.shape[0]  # 보통 1
        T = au_seq.shape[0] if B==1 else au_seq.shape[1]
        # runtime assert
        if det_seq.dim() == 2:
            assert det_seq.shape[0] == T, f"det_seq T={det_seq.shape[0]} != {T}"
        else:
            assert det_seq.shape[1] == T, f"det_seq T={det_seq.shape[1]} != {T}"

        # (1) 토큰화
        x_au  = self.proj_au(au_seq).view(B, T, -1)        # (B,T,D)
        x_det = self.proj_det(det_seq).view(B, T, -1)      # (B,T,D)
        # 타입 임베딩 부여
        type_emb = self.type_embedding[:, :2*T, :].clone()
        x_cat = torch.cat([x_au, x_det], dim=1) + type_emb # (B,2T,D)

        # (2) CLS & Pos
        cls_g = self.cls_g.repeat(B,1,1)
        cls_d = self.cls_d.repeat(B,1,1)
        x = torch.cat([cls_g, cls_d, x_cat], dim=1)        # (B,2+2T,D)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.dropout(x)

        # (3) 글로벌 메모리 y (기본 1토큰)
        # 2D로 들어오면 batch 차원만 붙여 (1,T,D)로 만든 다음,
        # 항상 시간축 평균으로 1토큰을 만듭니다. (T를 임베딩으로 절대 평탄화하지 않음!)
        if global_feature.dim() == 2:  # (T, D_glob) -> (1, T, D_glob)
            global_feature = global_feature.unsqueeze(0)
        elif global_feature.dim() == 3:  # (B, T, D_glob)
            pass
        else:
            raise ValueError(f"global_feature must be (T,D) or (B,T,D), got {tuple(global_feature.shape)}")

        y = self.FC_glob(global_feature)  # (B, T, 512)
        y = y.mean(dim=1, keepdim=True)  # (B, 1, 512) 설계 유지

        # (4) 트랜스포머
        x = self.transformer(x, y)                         # Self→Cross→FFN × depth

        # (5) 헤드 2개
        cls_g_out = x[:, 0]                                # [CLS_g]
        cls_d_out = x[:, 1]                                # [CLS_d]
        out_g = self.head_g(cls_g_out)
        out_d = self.head_d(cls_d_out)
        return out_g, out_d

    def _infer_det_in_dim(self):   # 필요하면 cfg로 치환
        return 256  # 예시: n_detail (환경에 맞춰 수정)
