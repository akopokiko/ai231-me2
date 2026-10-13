import torch
import torch.nn as nn
import torch.nn.functional as F

class BCResNetBlock(nn.Module):
    """
    Simplified Broadcasted Residual Network Block
    """
    def __init__(self, in_channels, out_channels, stride=(1, 1), dilation=(1, 1)):
        super().__init__()
        
        # 1. Pointwise convolution (transition)
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        # 2. Depthwise convolution over frequency
        self.dw_f = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=(stride[0], 1), padding=(2, 0), groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        # 3. Depthwise convolution over time
        self.dw_t = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=(1, stride[1]), padding=(0, 1), dilation=(1, dilation[1]), groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        # 4. Pointwise convolution
        self.pw = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if stride != (1,1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        
        x = self.transition(x)
        x = self.dw_f(x)
        x = self.dw_t(x)
        x = self.pw(x)
        
        return F.silu(x + res)

class BCResNet(nn.Module):
    """
    Multi-head BC-ResNet for Joint Intent Classification and Slot Filling
    (20 Intents including UNKNOWN, 19 Slots including No Slot)
    """
    def __init__(self, num_intents=20, num_slots=19):
        super().__init__()
        
        # Input shape: (Batch, 1, 64, 251)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU()
        )
        
        self.blocks = nn.Sequential(
            BCResNetBlock(16, 24, stride=(1, 1)),
            BCResNetBlock(24, 32, stride=(2, 2)),
            BCResNetBlock(32, 48, stride=(2, 2)),
            BCResNetBlock(48, 64, stride=(2, 2))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        # Shared temporal average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multi-head classifiers
        self.intent_head = nn.Linear(128, num_intents)
        self.slot_head = nn.Linear(128, num_slots)
        
    def forward(self, x):
        # x: (B, 1, 64, T)
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        intent_logits = self.intent_head(x)
        slot_logits = self.slot_head(x)
        
        return intent_logits, slot_logits

if __name__ == "__main__":
    model = BCResNet(num_intents=20, num_slots=19)
    x = torch.randn(2, 1, 64, 251)
    intents, slots = model(x)
    print("Intent output shape:", intents.shape, "(Expected: [2, 20])")
    print("Slot output shape:", slots.shape, "(Expected: [2, 19])")
