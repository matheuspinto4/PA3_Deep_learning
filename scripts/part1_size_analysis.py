"""Parte 1 (análise) — o professor erra mais com animais pequenos? E o aluno?

Usa as caixas do MegaDetector (scripts/part1_megadetector.py) para medir a
área do animal no test_public e compara, por faixa de tamanho: piso, professor
zero-shot, professor melhorado e o aluno escolhido. Testa também a hipótese do
woodchuck-ímã: se ele é o destino de imagens com pouca evidência, a fração de
previsões "woodchuck" do professor deve crescer quando o animal encolhe.
E mede o custo do center crop do professor: o ViT vê só o quadrado central
(perde 12,5% de cada lateral de uma foto 4:3), o aluno vê o quadro inteiro.

Tudo aqui é medida no test_public (diagnóstico), nada é selecionado nele.

Saídas em outputs/part1/: 11_tamanho_animal.png, size_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pa3.config import CLASSES, OUTPUTS_DIR
from pa3.data import get_cache, is_night, split_labels
from pa3.plotting import COLORS, INK_MUTED, plt

OUT = OUTPUTS_DIR / "part1"
MD_CONF = 0.2
CROP_MARGIN = 0.125  # center crop 224 de uma foto 4:3 redimensionada para altura 224: corta 12,5% de cada lado
BINS = [(0, 0.02, "<2%"), (0.02, 0.05, "2–5%"), (0.05, 0.15, "5–15%"), (0.15, 1.01, "≥15%")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows (cp1252) não imprime "≥"
    labels = split_labels("test_public")
    ids = sorted(labels)  # ordem de todos os arrays salvos (eval_dataset e professor)
    y = np.array([labels[i] for i in ids])
    md = json.loads((OUT / "megadetector" / "test_public.json").read_text())["detections"]

    # área do animal = maior caixa "animal" (conf >= 0.2); sem caixa = sem detecção
    # e fração da largura dessa caixa que cai fora do center crop do professor
    area, out_crop = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    for k, iid in enumerate(ids):
        b = [d[2:] for d in md.get(iid, []) if d[0] == "animal" and d[1] >= MD_CONF]
        if b:
            x1, y1, x2, y2 = max(b, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            area[k] = (x2 - x1) * (y2 - y1)
            inside = max(0.0, min(x2, 1 - CROP_MARGIN) - max(x1, CROP_MARGIN))
            out_crop[k] = 1 - inside / (x2 - x1) if x2 > x1 else 0.0
    night = is_night(get_cache("test_public").subset(ids))

    chosen = json.loads((OUT / "student_results.json").read_text())["chosen"]
    preds = {
        "piso (b)": np.load(OUTPUTS_DIR / "floor_runs" / "resnet18_b5_s0_test_logits.npy").argmax(1),
        "professor zero-shot": np.load(OUT / "teacher_test_probs_zs.npy").argmax(1),
        "professor melhorado": np.load(OUT / "teacher_test_probs_improved_b5.npy").argmax(1),
        "aluno": np.load(OUT / "students" / f"{chosen}_b5_s0_test_logits.npy").argmax(1),
    }
    colors = {"piso (b)": (COLORS["floor"], 1.0), "professor zero-shot": (COLORS["teacher"], 0.55),
              "professor melhorado": (COLORS["teacher"], 1.0), "aluno": (COLORS["student"], 1.0)}

    groups = [(lo, hi, lbl, (area >= lo) & (area < hi)) for lo, hi, lbl in BINS] + [(None, None, "sem detecção", np.isnan(area))]
    res = {"md_conf": MD_CONF, "student": chosen, "fraction_below_2pct": float(np.nanmean(area < 0.02)),
           "n_no_detection": int(np.isnan(area).sum()), "bins": {}}
    wc = CLASSES.index("woodchuck")
    for _, _, lbl, m in groups:
        res["bins"][lbl] = {"n": int(m.sum()), "frac_night": float(night[m].mean()) if m.any() else None,
                            "accuracy": {k: float((p[m] == y[m]).mean()) if m.any() else None for k, p in preds.items()},
                            "teacher_zs_pred_woodchuck": float((preds["professor zero-shot"][m] == wc).mean()) if m.any() else None,
                            "true_woodchuck": float((y[m] == wc).mean()) if m.any() else None}
    edge, center = out_crop >= 0.5, out_crop < 0.5  # NaN (sem detecção) fica fora dos dois
    res["center_crop"] = {lbl: {"n": int(m.sum()), "accuracy": {k: float((p[m] == y[m]).mean()) for k, p in preds.items()}}
                          for lbl, m in (("animal na borda", edge), ("animal no centro", center))}

    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(21, 5.6), gridspec_kw={"width_ratios": [1.6, 0.9, 1], "wspace": 0.25})
    w = 0.8 / len(preds)
    for j, (name, p) in enumerate(preds.items()):
        vals = [res["bins"][g[2]]["accuracy"][name] for g in groups]
        c, a = colors[name]
        bars = ax1.bar(np.arange(len(groups)) + (j - (len(preds) - 1) / 2) * w, vals, w * 0.92, color=c, alpha=a, label=name)
        ax1.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2)
    ax1.set_xticks(range(len(groups)), [f"{g[2]}\nn={res['bins'][g[2]]['n']}\n{res['bins'][g[2]]['frac_night']:.0%} noite" for g in groups])
    ax1.set_xlabel("área do animal no quadro (caixa do MegaDetector)")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("acurácia no test_public")
    ax1.set_title("Acurácia por tamanho do animal")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="x", visible=False)

    cc = res["center_crop"]
    w3 = 0.8 / len(preds)
    for j, name in enumerate(preds):
        c, a = colors[name]
        bars = ax3.bar(np.arange(2) + (j - (len(preds) - 1) / 2) * w3, [cc[g]["accuracy"][name] for g in cc], w3 * 0.92, color=c, alpha=a)
        ax3.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2)
    ax3.set_xticks(range(2), [f"{g}\nn={cc[g]['n']}" for g in cc])
    ax3.set_xlabel("≥50% da caixa fora do center crop do professor?")
    ax3.set_ylim(0, 1)
    ax3.set_title("O professor só vê o centro")
    ax3.grid(axis="x", visible=False)

    pw = [res["bins"][g[2]]["teacher_zs_pred_woodchuck"] for g in groups]
    tw = [res["bins"][g[2]]["true_woodchuck"] for g in groups]
    ax2.bar(np.arange(len(groups)) - 0.2, pw, 0.38, color=COLORS["teacher"], label="previsto woodchuck (professor zero-shot)")
    ax2.bar(np.arange(len(groups)) + 0.2, tw, 0.38, color=INK_MUTED, alpha=0.5, label="é woodchuck de verdade")
    ax2.set_xticks(range(len(groups)), [g[2].replace(" ", "\n") for g in groups])
    ax2.set_ylabel("fração das imagens da faixa")
    ax2.set_title("O woodchuck-ímã é efeito de animal pequeno?")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(axis="x", visible=False)
    fig.suptitle("Tamanho e posição do animal: piso, professor e aluno", fontweight="bold")
    fig.savefig(OUT / "11_tamanho_animal.png")
    plt.close(fig)
    (OUT / "size_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"animal < 2% do quadro em {res['fraction_below_2pct']:.0%} das imagens com detecção; sem detecção: {res['n_no_detection']}")
    for lbl, b in res["center_crop"].items():
        print(f"  {lbl:17s} n={b['n']:3d} " + " ".join(f"{k}={v:.2f}" for k, v in b["accuracy"].items()))
    for lbl, b in res["bins"].items():
        print(f"  {lbl:13s} n={b['n']:3d} " + " ".join(f"{k}={v:.2f}" for k, v in b["accuracy"].items() if v is not None)
              + f" | prof prevê woodchuck {b['teacher_zs_pred_woodchuck']:.0%} (verdadeiro {b['true_woodchuck']:.0%})")


if __name__ == "__main__":
    main()
