import torch
import torch.nn as nn
from basicsr.archs.NAFNet_arch import NAFBlock


class DeblurEnsemble(nn.Module):
    def __init__(self, in_channels=3, in_nums=4, out_channels=64, depth=4):
        super().__init__()
        self.encode = nn.Conv2d(in_channels=in_channels * in_nums, 
                                out_channels=out_channels, 
                                kernel_size=3, 
                                padding=1, 
                                stride=1, 
                                groups=1,
                                bias=True)
        self.ensemble = nn.Sequential(
                        *[NAFBlock(out_channels) for _ in range(depth)]
                    )
        self.decode = nn.Conv2d(in_channels=out_channels, 
                                out_channels=in_channels, 
                                kernel_size=3, 
                                padding=1, 
                                stride=1, 
                                groups=1,
                                bias=True)

        self.device = torch.device('cuda')

    def forward(self, x):
        self.to(self.device)
        x = x.to(self.device)

        x = self.encode(x)
        x = self.ensemble(x)
        x = self.decode(x)
        return x
