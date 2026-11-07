"""Estilo e gráficos reaproveitados entre as partes (pensados para slide)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pa3.config import CLASSES

# Paleta categórica validada (ordem fixa; a cor segue a entidade, nunca o rank).
COLORS = {
    "floor": "#2a78d6",  # piso / supervisionado
    "student": "#eb6834",  # aluno (Trilha C)
    "teacher": "#1baf7a",  # professor (foundation model)
    "scratch": "#6250d6",  # CNN do zero
    "ceiling": "#52514e",  # teto (linha de referência, cinza neutro)
}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def plot_confusion(cm: np.ndarray, title: str, path, normalize: bool = True) -> None:
    """Matriz de confusão normalizada por linha (recall por classe), com a
    contagem bruta escrita nas células não-nulas."""
    rows = cm.sum(axis=1, keepdims=True)
    norm = cm / np.maximum(rows, 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(CLASSES)), [f"{c} ({n})" for c, n in zip(CLASSES, rows.ravel())], fontsize=9)
    ax.set_xlabel("classe prevista")
    ax.set_ylabel("classe verdadeira (nº de imagens)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5, color="white" if norm[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="fração da linha (recall)")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)
