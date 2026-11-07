"""Mapas de saliência (Parte 3): Grad-CAM e oclusão, e quanto deles cai no animal.

- Grad-CAM na layer4 da ResNet-18: com entrada 224x320 o mapa nativo é 7x10
  (cada célula cobre 32x32 px), então ele é grosso por construção.
- Oclusão: cobre um quadrado (cinza = média do ImageNet, que vira 0 depois da
  normalização) e mede quanto a probabilidade da classe cai. Mais fino e não
  depende de gradiente, mas custa um forward por posição.
- `mass_in_box`: fração da saliência dentro da caixa do MegaDetector. Se o
  modelo olhasse para lugares aleatórios, essa fração seria igual à área da
  caixa; bem acima disso = olha para o animal; abaixo = olha para o fundo.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gradcam(model: nn.Module, x: torch.Tensor, target: int, layer: nn.Module | None = None) -> np.ndarray:
    """x: (1,3,H,W) normalizado. Devolve mapa (H,W) em [0,1]."""
    layer = layer or model.layer4
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        model.zero_grad()
        x = x.clone().requires_grad_(True)
        model(x)[0, target].backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


@torch.no_grad()
def occlusion(model: nn.Module, x: torch.Tensor, target: int, patch: int = 32, stride: int = 16, batch: int = 64) -> np.ndarray:
    """Queda da probabilidade de `target` ao tapar cada região. (H,W) em [0,1]."""
    model.eval()
    _, _, H, W = x.shape
    base = F.softmax(model(x), 1)[0, target].item()
    ys, xs = list(range(0, H - patch + 1, stride)), list(range(0, W - patch + 1, stride))
    coords = [(y, xx) for y in ys for xx in xs]
    drops = []
    for b in range(0, len(coords), batch):
        xb = x.repeat(len(coords[b : b + batch]), 1, 1, 1)
        for k, (y, xx) in enumerate(coords[b : b + batch]):
            xb[k, :, y : y + patch, xx : xx + patch] = 0.0
        drops.append(base - F.softmax(model(xb), 1)[:, target])
    drops = torch.cat(drops).clamp(min=0).numpy()
    heat, count = np.zeros((H, W)), np.zeros((H, W))
    for d, (y, xx) in zip(drops, coords):
        heat[y : y + patch, xx : xx + patch] += d
        count[y : y + patch, xx : xx + patch] += 1
    heat /= np.maximum(count, 1)
    return heat / heat.max() if heat.max() > 0 else heat


def box_mask(box_norm, shape) -> np.ndarray:
    """Máscara booleana (H,W) de uma caixa [x1,y1,x2,y2] normalizada (0-1)."""
    H, W = shape
    x1, y1, x2, y2 = box_norm
    m = np.zeros((H, W), dtype=bool)
    m[int(np.floor(y1 * H)) : int(np.ceil(y2 * H)), int(np.floor(x1 * W)) : int(np.ceil(x2 * W))] = True
    return m


def mass_in_box(sal: np.ndarray, boxes) -> dict:
    """Fração da saliência dentro da(s) caixa(s), a área que elas ocupam, e se
    o pico do mapa cai dentro (pointing game)."""
    m = np.zeros(sal.shape, dtype=bool)
    for b in boxes:
        m |= box_mask(b, sal.shape)
    total = sal.sum()
    peak = np.unravel_index(sal.argmax(), sal.shape)
    return {
        "mass_in_box": float(sal[m].sum() / total) if total > 0 else float("nan"),
        "box_area": float(m.mean()),
        "peak_in_box": bool(m[peak]),
    }
