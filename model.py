import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import DenseNetConfig


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate, memory_efficient):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate, growth_rate, 3, padding=1, bias=False
        )
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def _bottleneck(self, x):
        return self.conv1(F.relu(self.norm1(x)))

    def forward(self, x):
        if self.memory_efficient and x.requires_grad:
            out = checkpoint(self._bottleneck, x, use_reentrant=False)
        else:
            out = self._bottleneck(x)
        out = self.conv2(F.relu(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate, memory_efficient):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate,
                    bn_size,
                    drop_rate,
                    memory_efficient,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new = layer(torch.cat(features, dim=1))
            features.append(new)
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.norm(x))))


class DenseNet(nn.Module):
    def __init__(self, config: DenseNetConfig):
        super().__init__()
        self.config = config
        c = config.num_init_features

        self.stem = nn.Sequential(
            nn.Conv2d(config.num_canales, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        blocks = []
        for i, num_layers in enumerate(config.block_config):
            blocks.append(
                DenseBlock(
                    num_layers,
                    c,
                    config.growth_rate,
                    config.bn_size,
                    config.drop_rate,
                    config.ahorro_memoria,
                )
            )
            c += num_layers * config.growth_rate
            if i != len(config.block_config) - 1:
                out_c = int(c * config.compression)
                blocks.append(Transition(c, out_c))
                c = out_c
        self.features = nn.Sequential(*blocks)

        self.final_norm = nn.BatchNorm2d(c)
        self.classifier = nn.Linear(c, config.num_clases)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values):
        x = self.stem(pixel_values)
        x = self.features(x)
        x = F.relu(self.final_norm(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
