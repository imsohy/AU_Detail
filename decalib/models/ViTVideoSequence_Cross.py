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


class ViT(nn.Module):
    def __init__(self, *, num_classes, frames, dim, depth, heads, mlp_dim, patch_dim = 512, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., layer = 9):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # mid_channel = 128
        mid_channel = 512
        self.FC_glob =  nn.Linear(2048, mid_channel)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 27, p2 = 512),
            # nn.Conv2d(in_channels=1,out_channels= 1, kernel_size = 8,stride=1, dilation=2),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=1,out_channels= 1, kernel_size = 6,stride=2),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
            nn.LayerNorm(13824),
            nn.Linear(13824, mid_channel),
            nn.LayerNorm(mid_channel),
        )
        # self.glob_emb = nn.Linear(1024, mid_channel)

        self.pos_embedding = nn.Parameter(torch.randn(1, 4, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer =  Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.transformer_14 = Transformer(128*30, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight) # initalize

    def forward(self, feature, global_feature,layers = 14):
        x = self.to_patch_embedding(feature[:,None,...]).view(1,3,-1)
        y = torch.mean(self.FC_glob(global_feature), dim=0).view(1,1,-1)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1  d', b = 1)
        # print(x.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :4]
        x = self.dropout(x)
        x = self.transformer(x, y)

        x = x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x),
