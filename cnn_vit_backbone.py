import torch
import torch.nn as nn
from torchvision.models import resnet18
from timm.models.vision_transformer import Block
from timm.models.vision_transformer import vit_base_patch16_224

class CNNViTBackbone(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=6):
        super().__init__()
        cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(cnn.children())[:-2]) # 輸出 [B, 512, 7, 7]
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        # positional encoding
        self.pos_embed = None
        self.embed_dim = embed_dim

        # vit encoder blocks
        self.encoder = nn.Sequential(*[
            Block(embed_dim, num_heads) for _ in range(depth)
        ])
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        if self.pos_embed is None or self.pos_embed.shape[1] != H * W:
            self.pos_embed = nn.Parameter(
                torch.randn(1, H * W, self.embed_dim, device=x.device)
            )

        x = x + self.pos_embed
        x = self.encoder(x)
        return x