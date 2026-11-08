"""Parte 3 — Galeria de falhas e atalhos (modelo final = weights/modelo_final.pt).

1. Para TODAS as 811 imagens do test_public: Grad-CAM da classe prevista e a
   fração da saliência dentro da caixa do animal (MegaDetector). Isso responde
   "olhou para o animal ou para o fundo?" com números, não só com exemplos.
2. Três erros e cinco acertos, escolhidos por regras fixas (abaixo), cada um
   com imagem + caixa, Grad-CAM, oclusão, classe verdadeira/prevista/confiança
   e os fatos para o diagnóstico (noite/dia, tamanho do animal em pixels,
   saliência na caixa, top-3).

Regras de escolha (decididas antes de olhar as imagens):
  erros:   E1 o erro mais confiante;
           E2 o erro com o menor animal detectado;
           E3 o erro, com animal grande o bastante (>=3% do quadro), em que o
              modelo menos olhou para o animal (atalho de fundo).
  acertos: 3 acertos com a atenção mais concentrada no animal e 2 acertos com
           a atenção mais fora dele (acertou pelo motivo errado?), todos de
           classes diferentes e com animal >=2% do quadro.

Saídas em outputs/part3/: 01_erros.png, 02_acertos.png,
03_atencao_animal_vs_fundo.png, 00_candidatos_erros.png, gallery.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from pa3.config import CLASSES, IMG_H, IMG_W, OUTPUTS_DIR
from pa3.data import eval_dataset, is_night, split_labels
from pa3.plotting import COLORS, INK_MUTED, plt
from pa3.saliency import gradcam, mass_in_box, occlusion
from predict import load_model

OUT = OUTPUTS_DIR / "part3"
MD_CONF = 0.2  # caixas de "animal" do MegaDetector com confiança >= 0.2


def animal_boxes(md: dict, iid: str) -> list:
    return [d[2:] for d in md.get(iid, []) if d[0] == "animal" and d[1] >= MD_CONF]


def describe(i, ids, y, probs, night, boxes, sal_stats) -> dict:
    p = probs[i]
    top3 = np.argsort(p)[::-1][:3]
    b = boxes[i]
    if b:
        # maior caixa de animal, em pixels da entrada da rede (320 x 224)
        x1, y1, x2, y2 = max(b, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]))
        size_px = [round((x2 - x1) * IMG_W), round((y2 - y1) * IMG_H)]
    else:
        size_px = None
    return {
        "image_id": ids[i], "true": CLASSES[y[i]], "pred": CLASSES[int(p.argmax())], "conf": float(p.max()),
        "top3": [(CLASSES[k], round(float(p[k]), 3)) for k in top3],
        "night": bool(night[i]), "n_animal_boxes": len(b), "animal_px_wxh": size_px,
        **{k: sal_stats[i][k] for k in ("mass_in_box", "box_area", "peak_in_box")},
    }


def overlay(ax, img, sal, title=None):
    ax.imshow(img)
    ax.imshow(sal, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=10)


def draw_boxes(ax, boxes, color="#1baf7a"):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(plt.Rectangle((x1 * IMG_W, y1 * IMG_H), (x2 - x1) * IMG_W, (y2 - y1) * IMG_H, fill=False, ec=color, lw=2))


def gallery(rows: list[dict], images, model, x_of, boxes_of, path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.5 * len(rows)))
    fig.subplots_adjust(wspace=0.04, hspace=0.42, top=1 - 0.5 / len(rows) / 4.5 * 2, bottom=0.05)
    axes = np.atleast_2d(axes)
    for r, ax in zip(rows, axes):
        k = r["_idx"]
        target = CLASSES.index(r["pred"])
        cam = gradcam(model, x_of(k), target)
        occ = occlusion(model, x_of(k), target)
        ax[0].imshow(images[k])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].grid(False)
        draw_boxes(ax[0], boxes_of(k))
        ok = r["true"] == r["pred"]
        ax[0].set_title(f"verdadeiro: {r['true']}\nprevisto: {r['pred']} ({r['conf']:.2f})", fontsize=10,
                        color=INK_MUTED if ok else COLORS["student"])
        overlay(ax[1], images[k], cam, "Grad-CAM (classe prevista)")
        draw_boxes(ax[1], boxes_of(k))
        overlay(ax[2], images[k], occ, "Oclusão (queda da prob. ao tapar)")
        draw_boxes(ax[2], boxes_of(k))
        size = f"{r['animal_px_wxh'][0]}×{r['animal_px_wxh'][1]} px" if r["animal_px_wxh"] else "nenhum animal detectado"
        info = (f"{'noite (IR)' if r['night'] else 'dia'} · animal: {size} ({r['box_area']:.1%} do quadro)\n"
                f"saliência na caixa: {r['mass_in_box']:.0%} · top-3: " + ", ".join(f"{c} {p:.2f}" for c, p in r["top3"]))
        ax[0].text(0, IMG_H + 8, info, fontsize=8.5, va="top", color=INK_MUTED)
        r["occlusion_mass_in_box"] = mass_in_box(occ, boxes_of(k))["mass_in_box"] if boxes_of(k) else None
    fig.suptitle(title, fontweight="bold", fontsize=14)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", type=Path, default=ROOT / "weights" / "modelo_final.pt")
    ap.add_argument("--md", type=Path, default=OUTPUTS_DIR / "part1" / "megadetector" / "test_public.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    model, classes, hw = load_model(args.pesos)
    assert list(classes) == CLASSES and tuple(hw) == (IMG_H, IMG_W)
    md = json.loads(args.md.read_text())["detections"]

    ds = eval_dataset("test_public")
    ids = sorted(split_labels("test_public"))  # mesma ordem de eval_dataset
    images = ds.images  # uint8 224x320, o que a rede vê
    y = np.asarray(ds.labels)
    with torch.no_grad():
        logits = torch.cat([model(torch.stack([ds[i][0] for i in range(b, min(b + 64, len(ds)))])) for b in range(0, len(ds), 64)])
    probs = F.softmax(logits, 1).numpy()
    pred = probs.argmax(1)
    night = is_night(images)
    boxes = [animal_boxes(md, iid) for iid in ids]

    def x_of(k):
        return ds[k][0][None]

    print("Grad-CAM em todas as imagens...", flush=True)
    stats = []
    for k in range(len(ds)):
        cam = gradcam(model, x_of(k), int(pred[k]))
        stats.append(mass_in_box(cam, boxes[k]) if boxes[k] else {"mass_in_box": np.nan, "box_area": 0.0, "peak_in_box": False})
    mass = np.array([s["mass_in_box"] for s in stats])
    area = np.array([s["box_area"] for s in stats])
    peak = np.array([s["peak_in_box"] for s in stats])
    has = np.array([bool(b) for b in boxes])
    correct = pred == y
    info = [describe(k, ids, y, probs, night, boxes, stats) for k in range(len(ds))]
    for k, r in enumerate(info):
        r["_idx"] = k

    # --- 03: atenção no animal, acertos vs erros, por tamanho -----------------
    ratio = mass / np.maximum(area, 1e-6)  # 1 = tanto quanto o acaso
    bins = [(0, 0.02, "<2%"), (0.02, 0.05, "2–5%"), (0.05, 0.15, "5–15%"), (0.15, 1.01, "≥15%")]
    agg = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    for k, (lo, hi, lbl) in enumerate(bins):
        sel = has & (area >= lo) & (area < hi)
        for j, (grp, mask, c) in enumerate((("acertos", correct, COLORS["floor"]), ("erros", ~correct, COLORS["student"]))):
            s = sel & mask
            agg[f"{lbl}|{grp}"] = {"n": int(s.sum()), "mass_in_box": float(np.nanmean(mass[s])) if s.any() else None,
                                   "pointing": float(peak[s].mean()) if s.any() else None}
            if s.any():
                ax1.bar(k + (j - 0.5) * 0.38, np.nanmean(mass[s]), 0.36, color=c, label=grp if k == 0 else None)
                ax1.text(k + (j - 0.5) * 0.38, np.nanmean(mass[s]) + 0.01, f"n={s.sum()}", ha="center", fontsize=8, color=INK_MUTED)
        ax1.plot([k - 0.4, k + 0.4], [area[sel].mean()] * 2 if sel.any() else [0, 0], color=INK_MUTED, ls="--", lw=1.2,
                 label="área da caixa (acaso)" if k == 0 else None)
    ax1.set_xticks(range(len(bins)), [f"{b[2]}\n(n={int((has & (area >= b[0]) & (area < b[1])).sum())})" for b in bins])
    ax1.set_xlabel("área do animal no quadro")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("fração do Grad-CAM dentro da caixa do animal")
    ax1.set_title("Quanto da atenção cai no animal")
    ax1.legend(loc="upper left")
    ax1.grid(axis="x", visible=False)
    acc_bins = [float(correct[has & (area >= lo) & (area < hi)].mean()) for lo, hi, _ in bins]
    nodet = ~has
    bars = ax2.bar(range(len(bins) + 1), acc_bins + [float(correct[nodet].mean()) if nodet.any() else 0], 0.6,
                   color=[COLORS["student"]] * len(bins) + [INK_MUTED])
    ax2.bar_label(bars, fmt="%.2f", padding=2)
    ax2.set_xticks(range(len(bins) + 1), [b[2] for b in bins] + [f"sem detecção\n(n={int(nodet.sum())})"])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("acurácia do modelo final")
    ax2.set_title("Acurácia por tamanho do animal")
    ax2.grid(axis="x", visible=False)
    fig.suptitle("Modelo final no test_public: olha para o animal ou para o fundo?", fontweight="bold")
    fig.savefig(out / "03_atencao_animal_vs_fundo.png")
    plt.close(fig)

    # --- escolha dos exemplos (regras do docstring) ---------------------------
    wrong = [r for r in info if r["true"] != r["pred"]]
    used = set()

    def take(cands, key, n=1, distinct_class=False):
        out, seen = [], set()
        for r in sorted(cands, key=key):
            if r["image_id"] in used or (distinct_class and r["true"] in seen):
                continue
            out.append(r)
            used.add(r["image_id"])
            seen.add(r["true"])
            if len(out) == n:
                break
        return out

    errors = []
    errors += take(wrong, key=lambda r: -r["conf"])
    errors += take([r for r in wrong if r["n_animal_boxes"]], key=lambda r: r["box_area"])
    errors += take([r for r in wrong if r["n_animal_boxes"] and r["box_area"] >= 0.03], key=lambda r: r["mass_in_box"] / r["box_area"])
    hits_ok = [r for r in info if r["true"] == r["pred"] and r["n_animal_boxes"] and r["box_area"] >= 0.02]
    hits = take(hits_ok, key=lambda r: -(r["mass_in_box"] / r["box_area"]), n=3, distinct_class=True)
    seen_cls = {r["true"] for r in hits}
    hits += take([r for r in hits_ok if r["true"] not in seen_cls], key=lambda r: r["mass_in_box"] / r["box_area"], n=2, distinct_class=True)

    x_of_k, boxes_of = x_of, (lambda k: boxes[k])
    gallery(errors, images, model, x_of_k, boxes_of, out / "01_erros.png", "Três erros do modelo final")
    gallery(hits, images, model, x_of_k, boxes_of, out / "02_acertos.png", "Cinco acertos: animal ou fundo?")

    # folha de candidatos: 12 erros variados, para conferência humana
    cand = take(wrong, key=lambda r: -r["conf"], n=12)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    for r, ax in zip(cand, axes.ravel()):
        k = r["_idx"]
        overlay(ax, images[k], gradcam(model, x_of(k), CLASSES.index(r["pred"])),
                f"{r['true']} → {r['pred']} ({r['conf']:.2f})\n{'noite' if r['night'] else 'dia'} · caixa {r['box_area']:.1%} · na caixa {r['mass_in_box']:.0%}"
                if r["n_animal_boxes"] else f"{r['true']} → {r['pred']} ({r['conf']:.2f})\nsem animal detectado")
        draw_boxes(ax, boxes[k])
    fig.suptitle("Candidatos: os 12 erros mais confiantes (Grad-CAM da classe prevista)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "00_candidatos_erros.png")
    plt.close(fig)

    clean = lambda rs: [{k: v for k, v in r.items() if k != "_idx"} for r in rs]  # noqa: E731
    summary = {
        "n_with_animal_box": int(has.sum()),
        "mean_mass_in_box": {"acertos": float(np.nanmean(mass[has & correct])), "erros": float(np.nanmean(mass[has & ~correct]))},
        "mean_box_area": {"acertos": float(area[has & correct].mean()), "erros": float(area[has & ~correct].mean())},
        "pointing_game": {"acertos": float(peak[has & correct].mean()), "erros": float(peak[has & ~correct].mean())},
        "by_size": agg, "accuracy_by_size": dict(zip([b[2] for b in bins] + ["sem detecção"], acc_bins + [float(correct[nodet].mean()) if nodet.any() else None])),
        "errors": clean(errors), "hits": clean(hits), "candidates": clean(cand),
    }
    (out / "gallery.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("mean_mass_in_box", "mean_box_area", "pointing_game", "accuracy_by_size")}, indent=1))
    for grp in ("errors", "hits"):
        print(f"\n{grp}:")
        for r in summary[grp]:
            print(f"  {r['image_id']} {r['true']} -> {r['pred']} ({r['conf']:.2f}) | {'noite' if r['night'] else 'dia'} | "
                  f"animal {r['animal_px_wxh']} px, {r['box_area']:.1%} | Grad-CAM na caixa {r['mass_in_box']:.0%} | oclusão na caixa {r['occlusion_mass_in_box']}")


if __name__ == "__main__":
    main()
