"""Classifica uma pasta de imagens com o modelo entregue (PA3).

    python predict.py --imagens PASTA --saida preds.csv

Escreve um CSV com colunas image_id,label (image_id = nome do arquivo sem a
extensão; label = nome da classe como em classes.csv). Roda em CPU, sem
internet, e imprime o número de parâmetros do modelo.

Autocontido de propósito: não depende do pacote de dados nem do resto do
repositório. Os nomes das classes e a resolução de entrada estão salvos junto
com os pesos (weights/modelo_final.pt), então a máquina do monitor só precisa
de torch, torchvision e Pillow.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "modelo_final.pt"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = resnet18(weights=None, num_classes=len(ck["classes"]))  # arquitetura só, nenhum download
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["classes"], tuple(ck["input_hw"])


def preprocess(path: Path, hw: tuple[int, int]) -> np.ndarray:
    """Igual ao treino: quadro inteiro (sem crop) redimensionado para H x W
    com PIL bilinear, depois normalização do ImageNet."""
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagens", required=True, type=Path, help="pasta com as imagens")
    ap.add_argument("--saida", required=True, type=Path, help="CSV de saída (image_id,label)")
    ap.add_argument("--pesos", type=Path, default=DEFAULT_WEIGHTS, help="arquivo de pesos (padrão: weights/modelo_final.pt)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, classes, hw = load_model(args.pesos)
    print(f"parâmetros do modelo: {sum(p.numel() for p in model.parameters()):,}")

    paths = sorted(p for p in args.imagens.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"nenhuma imagem encontrada em {args.imagens}")
    t0, rows = time.time(), []
    for b in range(0, len(paths), args.batch):
        chunk = paths[b : b + args.batch]
        x = torch.from_numpy(np.stack([preprocess(p, hw) for p in chunk]))
        pred = model(x).argmax(1).tolist()
        rows += [(p.stem, classes[k]) for p, k in zip(chunk, pred)]
        print(f"  {len(rows)}/{len(paths)}", end="\r", flush=True)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.saida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label"])
        w.writerows(rows)
    print(f"\n{len(rows)} imagens classificadas em {time.time() - t0:.1f}s (CPU) -> {args.saida}")


if __name__ == "__main__":
    main()
