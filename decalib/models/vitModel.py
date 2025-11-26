import torch
import torch.nn as nn
# from vit_pytorch import ViT
from .My_ViT import ViT
class ViTEncoder(nn.Module):
    # def __init__(self, image_size = 224, patch_size = 32, num_classes = 236, dim = 1024,
    def __init__(self, num_classes = 236, dim = 512,
                depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1):
        super(ViTEncoder, self).__init__()

        self.vit = ViT(
                num_feature= num_classes,
                dim = dim,
                depth = depth,
                heads = heads,
                mlp_dim = mlp_dim,
                dropout = 0.1,
                emb_dropout = 0.1
            )

# img = torch.randn(1, 3, 256, 256)


    def forward(self, inputs):
        # x = torch.cat((x, x, x), 1)
        preds  = self.vit(inputs)
        return preds