import torch
import torch.nn as nn
from torchvision.models import resnet18
from cnn_attention import CBAM  # 確保你已建立 cnn_attention.py 並放入 CBAM 定義

class CNN_CBAM_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # [B, 512, 7, 7]
        self.cbam = CBAM(512)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        # return x  # [B, 512, H, W]
        return {"features": x}