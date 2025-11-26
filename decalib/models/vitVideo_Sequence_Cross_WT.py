import torch
import torch.nn as nn
# from vit_pytorch import ViT
from .ViTVideoSequence_Cross_WT import ViT


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
