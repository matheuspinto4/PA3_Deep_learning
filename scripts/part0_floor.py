"""Parte 0 — O piso.

Supervisionado só com rotulos_5.csv (105 imagens), em duas versões:
  (a) CNN pequena treinada do zero;
  (b) ResNet-18 ImageNet com fine-tuning (o backbone do resto do PA).
Época escolhida pelo F1 macro no val; test_public só para reportar.

Gera em outputs/part0/:
  01_piso_f1_acuracia.png        F1 macro e acurácia de (a) e (b) + linha do teto
  02_matriz_confusao_resnet18.png matriz de confusão de (b) no test_public
  03_curva_resnet18.png          curva de treino e val de (b) (onde começa a decorar)
  03b_curva_small_cnn.png        a mesma curva para (a), para comparação
  04_f1_por_classe.png           F1 por classe de (a) e (b)
  results.json                   todos os números (inclui histórico por época)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pa3.config import CEILING_F1, CLASSES, OUTPUTS_DIR
from pa3.floor import train_floor
from pa3.metrics import confusion
from pa3.plotting import COLORS, INK_MUTED, plot_confusion, plt

OUT = OUTPUTS_DIR / "part0"
OUT.mkdir(parents=True, exist_ok=True)

ARCHS = {"small_cnn": "(a) CNN pequena do zero", "resnet18": "(b) ResNet-18 ImageNet + fine-tuning"}


def memorization_epochs(history: list[dict]) -> dict:
    """Dois marcadores objetivos de "começou a decorar":
    - primeira época em que a rede acerta ~todo o treino SEM aumentação;
    - época de menor perda no val (depois dela a perda de val só sobe enquanto
      a de treino continua caindo: o ganho no treino deixou de generalizar)."""
    fit_ep = next((h["epoch"] for h in history if h["train_acc"] >= 0.99), None)
    min_val_ep = min(history, key=lambda h: h["val_loss"])["epoch"]
    return {"train_acc_99_epoch": fit_ep, "min_val_loss_epoch": min_val_ep}


def plot_curve(res: dict, title: str, path: Path) -> None:
    h = res["history"]
    ep = [r["epoch"] for r in h]
    marks = res["memorization"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

    ax1.plot(ep, [r["train_loss"] for r in h], color=COLORS["floor"], label="treino (sem aumentação)")
    ax1.plot(ep, [r["train_loss_aug"] for r in h], color=COLORS["floor"], ls=":", lw=1.5, label="treino (com aumentação)")
    ax1.plot(ep, [r["val_loss"] for r in h], color=COLORS["student"], label="val")
    ax1.set_xlabel("época")
    ax1.set_ylabel("entropia cruzada")
    ax1.set_title("Perda")

    ax2.plot(ep, [r["train_f1"] for r in h], color=COLORS["floor"], label="treino (sem aumentação)")
    ax2.plot(ep, [r["val_f1"] for r in h], color=COLORS["student"], label="val")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("época")
    ax2.set_ylabel("F1 macro")
    ax2.set_title("F1 macro")

    for ax in (ax1, ax2):
        if marks["train_acc_99_epoch"]:
            ax.axvline(marks["train_acc_99_epoch"], color=INK_MUTED, ls="--", lw=1)
        ax.axvline(marks["min_val_loss_epoch"], color=COLORS["student"], ls="--", lw=1)
        ax.axvline(res["best_epoch"], color=COLORS["teacher"], ls="-", lw=1)
    y_top = ax2.get_ylim()[1]
    if marks["train_acc_99_epoch"]:
        ax2.text(marks["train_acc_99_epoch"], y_top, " treino ≥99%", color=INK_MUTED, fontsize=9, va="top")
    ax2.text(marks["min_val_loss_epoch"], 0.04, " mín. perda val", color=COLORS["student"], fontsize=9)
    ax2.text(res["best_epoch"], 0.12, f" melhor F1 val (ép. {res['best_epoch']})", color=COLORS["teacher"], fontsize=9)
    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    fig.suptitle(title, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def plot_summary(results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    metrics = [("test_f1_macro", "F1 macro"), ("test_accuracy", "acurácia")]
    width = 0.36
    for k, (arch, label) in enumerate(ARCHS.items()):
        vals = [results[arch][m] for m, _ in metrics]
        xs = np.arange(len(metrics)) + (k - 0.5) * width
        color = COLORS["scratch"] if arch == "small_cnn" else COLORS["floor"]
        bars = ax.bar(xs, vals, width * 0.94, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.axhline(CEILING_F1, color=COLORS["ceiling"], ls="--", lw=1.5)
    ax.text(1.55, CEILING_F1 + 0.01, f"teto F1 = {CEILING_F1}", color=COLORS["ceiling"], ha="right", fontsize=9)
    ax.set_xticks(range(len(metrics)), [m for _, m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Piso no test_public — 5 rótulos por classe (105 imagens)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_per_class(results: dict, path: Path) -> None:
    order = np.argsort(results["resnet18"]["test_f1_per_class"])
    y = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(9, 8))
    h = 0.4
    for k, arch in enumerate(ARCHS):
        vals = np.array(results[arch]["test_f1_per_class"])[order]
        color = COLORS["scratch"] if arch == "small_cnn" else COLORS["floor"]
        ax.barh(y + (k - 0.5) * h, vals, h * 0.92, color=color, label=ARCHS[arch])
    ax.set_yticks(y, [CLASSES[i] for i in order], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 no test_public")
    ax.set_title("F1 por classe do piso (ordenado pela ResNet-18)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def main():
    results = {}
    for arch in ARCHS:
        print(f"\n=== {ARCHS[arch]} ===")
        out = train_floor(arch, budget=5, seed=0)
        res = out["result"]
        res["memorization"] = memorization_epochs(res["history"])
        results[arch] = res
        if arch == "resnet18":
            cm = confusion(out["test_labels"], out["test_logits"].argmax(1))
            res["confusion_matrix"] = cm.tolist()
            plot_confusion(
                cm,
                f"Piso (b) ResNet-18 + fine-tuning, 5 rótulos/classe — test_public\n"
                f"F1 macro {res['test_f1_macro']:.3f} · acurácia {res['test_accuracy']:.3f}",
                OUT / "02_matriz_confusao_resnet18.png",
            )
        plot_curve(res, f"Curva de treino e validação — {ARCHS[arch]}", OUT / ("03_curva_resnet18.png" if arch == "resnet18" else "03b_curva_small_cnn.png"))
        (OUT / "results.json").write_text(json.dumps(results, indent=2))

    plot_summary(results, OUT / "01_piso_f1_acuracia.png")
    plot_per_class(results, OUT / "04_f1_por_classe.png")

    print("\n| versão | parâmetros | melhor época (val) | F1 val | F1 macro test | acurácia test |")
    print("|---|---|---|---|---|---|")
    for arch, label in ARCHS.items():
        r = results[arch]
        print(f"| {label} | {r['params']:,} | {r['best_epoch']} | {r['best_val_f1']:.3f} | {r['test_f1_macro']:.3f} | {r['test_accuracy']:.3f} |")
    for arch in ARCHS:
        print(arch, "marcadores de memorização:", results[arch]["memorization"])


if __name__ == "__main__":
    main()
