from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 48) -> None:
        super().__init__()
        self.down1 = ConvBlock(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))

        u3 = self.up3(b)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        return self.dec1(u1)


class DiscreteRGBUNet(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 48, rgb_bins: int = 256) -> None:
        super().__init__()
        self.rgb_bins = rgb_bins
        self.backbone = UNetBackbone(in_channels=in_channels, base=base)
        self.out = nn.Conv2d(base, 3 * rgb_bins, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.out(features)
        batch, _, height, width = logits.shape
        return logits.view(batch, 3, self.rgb_bins, height, width)


def rgb_logits_to_uint8(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=2).to(torch.uint8)


def rgb_uint8_to_float(rgb: torch.Tensor) -> torch.Tensor:
    return rgb.to(torch.float32) / 255.0


def compose_rgba(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.cat([rgb, alpha], dim=1)
