import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, 1
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = x + identity
        x = self.relu(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=True):
        super(EncoderBlock, self).__init__()

        if use_residual:
            self.conv_block = ResidualConvBlock(in_channels, out_channels)
        else:
            self.conv_block = ConvBlock(in_channels, out_channels)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv_block(x)
        pooled = self.max_pool(skip)
        return pooled, skip

class Encoder(nn.Module):
    def __init__(self, use_residual=True):
        super().__init__()

        self.encoder_block_1 = EncoderBlock(3, 32, use_residual)
        self.encoder_block_2 = EncoderBlock(32, 64, use_residual)
        self.encoder_block_3 = EncoderBlock(64, 128, use_residual)
        self.encoder_block_4 = EncoderBlock(128, 256, use_residual)

    def forward(self, x):
        self.skips = []

        pooled1, skip1 = self.encoder_block_1(x)
        self.skips.append(skip1)

        pooled2, skip2 = self.encoder_block_2(pooled1)
        self.skips.append(skip2)

        pooled3, skip3 = self.encoder_block_3(pooled2)
        self.skips.append(skip3)

        pooled4, skip4 = self.encoder_block_4(pooled3)
        self.skips.append(skip4)

        return pooled4, self.skips

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=True):
        super().__init__()

        if use_residual:
            self.conv_block = ResidualConvBlock(in_channels, out_channels)
        else:
            self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv_block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_residual=True):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

        if use_residual:
            self.conv_block = ResidualConvBlock(skip_channels + out_channels, out_channels)
        else:
            self.conv_block = ConvBlock(skip_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, use_residual=True):
        super().__init__()

        self.decoder_block_1 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
            use_residual=use_residual
        )

        self.decoder_block_2 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            use_residual=use_residual
        )

        self.decoder_block_3 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
            use_residual=use_residual
        )
        
        self.decoder_block_4 = DecoderBlock(
            in_channels=64,
            skip_channels=32,
            out_channels=32,
            use_residual=use_residual
        )

    def forward(self, x, skips):
        x = self.decoder_block_1(x, skips[3])
        x = self.decoder_block_2(x, skips[2])
        x = self.decoder_block_3(x, skips[1])
        x = self.decoder_block_4(x, skips[0])
        return x

class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.one_one_conv = nn.Conv2d(
            in_channels = 32,
            out_channels = 1,
            kernel_size = 1
            )

    def forward(self, x):
        return self.one_one_conv(x)

class BackgroundRemoval(nn.Module):
    def __init__(self, use_residual=True):
        super().__init__()
        self.encoder = Encoder(use_residual)
        self.bottleneck = Bottleneck(256, 512, use_residual)
        self.decoder = Decoder(use_residual)
        self.seg_head = SegmentationHead()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.dropout(x)
        x = self.decoder(x, skips)
        x = self.seg_head(x)
        return x
