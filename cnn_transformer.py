import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2
from timm.models.vision_transformer import Block

class CNNTransformer(nn.Module):
    def __init__ (self, embed_dim=512, num_heads=8, depth=4):
        super().__init__()
        cnn = wide_resnet50_2(pretrained=True)
        self.cnn = nn.Sequential(*list(cnn.children())[:-2]) # 去掉 avgpool 和 fc

        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim)) # 假設輸出為 7x7

        self.encoder = nn.Sequential(*[
            Block(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
    
    def forward(self, x):
        feat = self.cnn(x)                        # [B, 2048, 7, 7]
        feat = self.proj(feat)                    # [B, embed_dim, 7, 7]
        B, C, H, W = feat.shape
        feat = feat.flatten(2).permute(0, 2, 1)   # [B, 49, C]
        feat = feat + self.pos_embed[:, :feat.size(1), :]
        feat = self.encoder(feat)                 # [B, 49, C]
        return feat.mean(dim=1)                   # [B, C]