"""Leitura dos CSVs, cache de imagens pré-redimensionadas e Dataset.

Decodificar ~6.5k JPEGs de 512px a cada época custaria mais que o próprio
forward em CPU. Por isso cada split é redimensionado UMA vez para IMG_H x IMG_W
e salvo em `cache/<split>.npy` (uint8, N x H x W x 3), junto com a ordem dos
ids em `cache/<split>_ids.txt`. Treino e avaliação leem só do cache.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from pa3.config import CACHE_DIR, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD, IMAGES_DIR, IMG_H, IMG_W, LABELS_DIR

SPLITS = ("train", "val", "test_public")


# ----------------------------------------------------------------------------
# Rótulos
# ----------------------------------------------------------------------------


def read_label_csv(path: Path) -> dict[str, int]:
    with open(path, newline="") as f:
        return {r["image_id"]: CLASS_TO_IDX[r["label"]] for r in csv.DictReader(f)}


def budget_labels(budget: int) -> dict[str, int]:
    """Rótulos do orçamento de `budget` por classe (subconjunto do train)."""
    return read_label_csv(LABELS_DIR / f"rotulos_{budget}.csv")


def split_labels(split: str) -> dict[str, int]:
    """Rótulos de val ou test_public."""
    return read_label_csv(LABELS_DIR / f"{split}.csv")


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------


def load_and_resize(path: Path, size: tuple[int, int] = (IMG_H, IMG_W)) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def build_cache(split: str, image_dir: Path | None = None, overwrite: bool = False) -> None:
    image_dir = image_dir or IMAGES_DIR / split
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arr_path, ids_path = CACHE_DIR / f"{split}.npy", CACHE_DIR / f"{split}_ids.txt"
    if arr_path.exists() and ids_path.exists() and not overwrite:
        return
    paths = sorted(image_dir.glob("*.jpg"))
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8, shape=(len(paths), IMG_H, IMG_W, 3))
    for i, p in enumerate(paths):
        arr[i] = load_and_resize(p)
    arr.flush()
    ids_path.write_text("\n".join(p.stem for p in paths))


class SplitCache:
    """Imagens de um split inteiro em memória, indexáveis por image_id."""

    def __init__(self, split: str):
        self.split = split
        self.images = np.load(CACHE_DIR / f"{split}.npy")
        self.ids = (CACHE_DIR / f"{split}_ids.txt").read_text().split("\n")
        self.index = {iid: i for i, iid in enumerate(self.ids)}

    def subset(self, ids: list[str]) -> np.ndarray:
        return self.images[[self.index[i] for i in ids]]


_CACHES: dict[str, SplitCache] = {}


def get_cache(split: str) -> SplitCache:
    if split not in _CACHES:
        _CACHES[split] = SplitCache(split)
    return _CACHES[split]


def colorfulness(images: np.ndarray) -> np.ndarray:
    """Diferença média entre canais por imagem (0 = perfeitamente cinza)."""
    x = images.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return ((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3).mean(axis=(1, 2))


NIGHT_COLORFULNESS = 2.0  # foto IR noturna: os 3 canais quase iguais


def is_night(images: np.ndarray) -> np.ndarray:
    return colorfulness(images) < NIGHT_COLORFULNESS


# ----------------------------------------------------------------------------
# Transformações
# ----------------------------------------------------------------------------

_normalize = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform() -> v2.Compose:
    """Aumentação de fine-tuning, escolhida pensando em câmera trap.

    - flip horizontal: animais andam para os dois lados;
    - crop leve (>=70% da área, aspecto perto do original): um crop agressivo
      tiraria do quadro o animal pequeno, e o rótulo deixaria de ser verdade;
    - jitter de cor + escala de cinza aleatória (p=0.2): parte das fotos é
      noturna em infravermelho, sem cor -- a rede não pode depender da cor.
    """
    return v2.Compose(
        [
            v2.RandomResizedCrop((IMG_H, IMG_W), scale=(0.7, 1.0), ratio=(1.25, 1.6), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            v2.RandomGrayscale(p=0.2),
            _normalize,
        ]
    )


def eval_transform() -> v2.Compose:
    return _normalize


class ImageArrayDataset(Dataset):
    """(N,H,W,3) uint8 + rótulos -> (tensor normalizado, rótulo).

    `labels` pode ser inteiro (classe dura), vetor de probabilidades (rótulo
    suave) ou -1 (sem rótulo).
    """

    def __init__(self, images: np.ndarray, labels=None, transform=None):
        self.images = images
        self.labels = labels if labels is not None else np.full(len(images), -1)
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.images[i])).permute(2, 0, 1)
        return self.transform(x), self.labels[i]


def labeled_dataset(labels: dict[str, int], split: str, transform=None) -> ImageArrayDataset:
    ids = sorted(labels)
    cache = get_cache(split)
    return ImageArrayDataset(cache.subset(ids), np.array([labels[i] for i in ids]), transform)


def budget_dataset(budget: int, transform=None) -> ImageArrayDataset:
    return labeled_dataset(budget_labels(budget), "train", transform)


def eval_dataset(split: str) -> ImageArrayDataset:
    return labeled_dataset(split_labels(split), split)
