import torch
import torch.nn as nn
# from vit_pytorch import ViT
from .My_ViT_Video import ViT
class ViTEncoderV(nn.Module):
    # def __init__(self, image_size = 224, patch_size = 32, num_classes = 236, dim = 1024,
    def __init__(self,num_features = 512, dim = 1024,
                depth = 2, heads = 4, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1):
        super(ViTEncoderV, self).__init__()
  
        self.vit = ViT(
                num_classes = num_features,
                dim = dim,
                depth = depth,
                heads = heads,
                mlp_dim = mlp_dim,
                dropout = 0.1,
                emb_dropout = 0.1,
            )

# img = torch.randn(1, 3, 256, 256)


    def forward(self, inputs, i):
        # x = torch.squeeze(inputs,2)
        x = torch.unsqueeze(inputs, 0)
        # print(x.shape)
        preds = self.vit(x, i)
        return preds[0]