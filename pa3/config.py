"""Caminhos, classes e constantes compartilhadas por todo o PA3."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch

# Treino roda em GPU quando há (Colab) e em CPU caso contrário. O modelo
# entregue (predict.py) é sempre avaliado em CPU, como o enunciado exige.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# No Windows os workers do DataLoader custam mais do que ajudam (spawn);
# no Colab (Linux, fork) os workers fazem a aumentação em paralelo à GPU.
NUM_WORKERS = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pa3_dados" / "pa3_dados"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
CACHE_DIR = ROOT / "cache"
OUTPUTS_DIR = ROOT / "outputs"
CKPT_DIR = ROOT / "checkpoints"

# Entrada da rede: quadro inteiro redimensionado para 224x320 (altura x largura).
# Sem center crop: num quarto das imagens o animal ocupa <2% do quadro e pode
# estar na borda -- um crop quadrado jogaria ele fora. 320/224 = 1.43 fica perto
# da razão de aspecto típica das imagens (512x367 = 1.40, 512x359 = 1.43); as
# mais largas (512x270 = 1.90) sofrem um achatamento horizontal leve.
IMG_H, IMG_W = 224, 320

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Teto de referência do enunciado: ResNet-18 ImageNet + fine-tuning com TODOS
# os rótulos do train, época escolhida no val (F1 macro no test_public).
CEILING_F1 = 0.686

BUDGETS = (1, 5, 20)


def load_classes() -> list[str]:
    with open(DATA_DIR / "classes.csv", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["idx"]))
    return [r["label"] for r in rows]


CLASSES = load_classes()
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
