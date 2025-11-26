import torch
import torch.nn as nn
# from vit_pytorch import ViT
from .ViTVideoSequence_Cross_WT import ViT
from .ViTVideoSequence_Cross_WT_detail import ViT_Multi

#new!!!!!
class ViTEncoderSequeMulti(nn.Module):
    def __init__(self, num_features_g, num_features_d, frames=3, dim=512, depth=6, heads=4, mlp_dim=1024):
        super().__init__()
        self.vit = ViT_Multi(num_classes_g=num_features_g,
                             num_classes_d=num_features_d,
                             frames=frames,
                             dim=dim,
                             depth=depth,
                             heads=heads,
                             mlp_dim=mlp_dim,
                             det_in_dim=num_features_d
                             )

    def forward(self, afn_flat, detail_seq, global_feature):
        # afn_flat: (T, au_in_dim)  ← 아래 encode에서 준비해 줌
        return self.vit(afn_flat, detail_seq, global_feature)  # (out_g, out_d)


class ViTEncoderSeque(nn.Module):
    # def __init__(self, image_size = 224, patch_size = 32, num_classes = 236, dim = 1024,
    def __init__(self, num_features=236, frames=3, dim=512,#dim=128*29,
                 depth=6, heads=4, mlp_dim=1024, dropout=0.1, emb_dropout=0.1):
        super(ViTEncoderSeque, self).__init__()

        self.vit = ViT(
            num_classes=num_features,
            frames=frames,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
        )

    # img = torch.randn(1, 3, 256, 256)

    def forward(self, inputs, global_feature):
        # x = torch.squeeze(inputs,2)
        # x = torch.unsqueeze(inputs, 0)
        # print(x.shape)
        # print(inputs.shape, global_feature.shape)
        preds = self.vit(inputs, global_feature)
        return preds[0]
