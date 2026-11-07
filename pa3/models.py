"""Arquiteturas: CNN pequena do zero (Parte 0a) e ResNet-18 ImageNet (todo o resto)."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from pa3.config import NUM_CLASSES

MAX_PARAMS = 12_000_000


def count_params(model: nn.Module) -> int:
    """Contagem exatamente como o enunciado define."""
    return sum(p.numel() for p in model.parameters())


def _conv_bn(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """CNN pequena treinada do zero (~1.2M parâmetros).

    Stem com stride 2 (224x320 -> 112x160) para caber em CPU, depois 4 estágios
    de duas convs 3x3 + max-pool, global average pooling e dropout antes da
    camada linear. GAP em vez de flatten: com 105 imagens, uma camada densa
    sobre o mapa inteiro teria milhões de pesos para decorar o fundo.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, widths=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers = [_conv_bn(3, widths[0], stride=2)]
        c_in = widths[0]
        for w in widths:
            layers += [_conv_bn(c_in, w), _conv_bn(w, w), nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 com pesos de classificação supervisionada do ImageNet (permitido
    pelo enunciado) e cabeça nova de 21 classes. 11.19M parâmetros < 12M."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "small_cnn":
        return SmallCNN()
    if arch == "resnet18":
        return build_resnet18(pretrained)
    raise ValueError(arch)


def param_groups(model: nn.Module, arch: str, lr: float, head_lr_mult: float) -> list[dict]:
    """Na ResNet pré-treinada, a cabeça nova aprende mais rápido que o backbone
    (que já sabe extrair features e só precisa de ajuste fino)."""
    if arch != "resnet18" or head_lr_mult == 1.0:
        return [{"params": model.parameters(), "lr": lr}]
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": body, "lr": lr}, {"params": head, "lr": lr * head_lr_mult}]
