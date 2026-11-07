"""Parte 2 — A curva de orçamento.

Piso (Parte 0b) e o método da Trilha C (o aluno escolhido no val na Parte 1)
com 1, 5 e 20 rótulos por classe. No orçamento de 5, 3 seeds (média ± desvio);
nos outros, 1 seed. O professor entra como referência: o zero-shot não usa
rótulo nenhum (linha horizontal), e o professor melhorado usa os rótulos do
orçamento (um ponto por orçamento; determinístico, sem seed).

Toda rodada é retomável (ver pa3/floor.py e pa3/student.py): rodar de novo só
treina o que falta. Pressupõe outputs/part1/student_results.json (Parte 1).

Saídas:
  curva_orcamento.csv                       (raiz do repo, entregável; cópia em outputs/part2/)
  outputs/part2/01_curva_orcamento.png      o gráfico obrigatório
  outputs/part2/02_ganho_por_classe_b5.png  ganho do método sobre o piso por classe
  outputs/part2/results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pa3.config import BUDGETS, CEILING_F1, CLASSES, NUM_CLASSES, OUTPUTS_DIR
from pa3.floor import train_floor
from pa3.plotting import COLORS, INK_MUTED, plt
from pa3.student import train_student

OUT = OUTPUTS_DIR / "part2"
SEEDS = {1: [0], 5: [0, 1, 2], 20: [0]}


def run_all(method: str) -> list[dict]:
    rows = []
    for budget in BUDGETS:
        for seed in SEEDS[budget]:
            f = train_floor("resnet18", budget=budget, seed=seed)["result"]
            rows.append(dict(metodo="piso", orcamento=budget, seed=seed, f1_macro=f["test_f1_macro"], acuracia=f["test_accuracy"],
                             f1_per_class=f["test_f1_per_class"], best_epoch=f["best_epoch"], val_f1=f["best_val_f1"]))
            s = train_student(method, budget=budget, seed=seed)
            rows.append(dict(metodo=f"aluno ({method})", orcamento=budget, seed=seed, f1_macro=s["test_f1_macro"], acuracia=s["test_accuracy"],
                             f1_per_class=s["test_f1_per_class"], best_epoch=s["best_epoch"], val_f1=s["best_val_f1"]))
    return rows


def teacher_rows() -> list[dict]:
    t = json.loads((OUTPUTS_DIR / "part1" / "teacher_results.json").read_text())
    rows = []
    for budget in BUDGETS:
        rows.append(dict(metodo="professor zero-shot", orcamento=budget, seed="-", f1_macro=t["teacher_zs"]["f1_macro"],
                         acuracia=t["teacher_zs"]["accuracy"]))
        b = t[f"budget_{budget}"]
        rows.append(dict(metodo="professor melhorado", orcamento=budget, seed="-", f1_macro=b["improved_test_f1_macro"],
                         acuracia=b["improved_test_accuracy"]))
    return rows


def summarize(rows: list[dict], metodo: str) -> dict:
    out = {}
    for budget in BUDGETS:
        v = [r["f1_macro"] for r in rows if r["metodo"] == metodo and r["orcamento"] == budget]
        a = [r["acuracia"] for r in rows if r["metodo"] == metodo and r["orcamento"] == budget]
        out[budget] = {"f1_mean": float(np.mean(v)), "f1_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                       "acc_mean": float(np.mean(a)), "n_seeds": len(v)}
    return out


def plot_curve(summary: dict, method_label: str, path: Path) -> None:
    x = np.array([b * NUM_CLASSES for b in BUDGETS])
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    styles = [
        ("piso", "piso (ResNet-18, só rótulos)", COLORS["floor"], "o", "-"),
        (method_label, "aluno (Trilha C)", COLORS["student"], "o", "-"),
        ("professor melhorado", "professor melhorado (BioCLIP 2 + rótulos)", COLORS["teacher"], "s", "--"),
    ]
    # referências na legenda (não em texto solto): a faixa 0,6-0,75 fica cheia
    zs = summary["professor zero-shot"][BUDGETS[0]]["f1_mean"]
    ax.axhline(CEILING_F1, color=COLORS["ceiling"], ls="--", lw=1.5, label=f"teto: todos os rótulos do train ({CEILING_F1})")
    ax.axhline(zs, color=COLORS["teacher"], ls=":", lw=1.5, label=f"professor zero-shot, 0 rótulos ({zs:.3f})")
    offsets = {"piso": (10, -14), "professor melhorado": (10, 8)}
    for key, lbl, c, mk, ls in styles:
        m = np.array([summary[key][b]["f1_mean"] for b in BUDGETS])
        e = np.array([summary[key][b]["f1_std"] for b in BUDGETS])
        ax.errorbar(x, m, yerr=e, color=c, marker=mk, ms=8, ls=ls, lw=2, capsize=5, label=lbl, mec="white", mew=1.5, zorder=3)
        for xi, mi, b in zip(x, m, BUDGETS):
            txt = f"{mi:.3f}" + (f" ± {summary[key][b]['f1_std']:.3f}" if summary[key][b]["n_seeds"] > 1 else "")
            ax.annotate(txt, (xi, mi), textcoords="offset points", xytext=offsets.get(key, (10, -3)), fontsize=8.5, color=c)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{b}/classe\n({b * NUM_CLASSES})" for b in BUDGETS])
    ax.minorticks_off()
    ax.set_xlim(x[0] * 0.75, x[-1] * 1.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("número de rótulos (escala log)")
    ax.set_ylabel("F1 macro no test_public")
    ax.set_title("Curva de orçamento")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_gain(rows: list[dict], method_label: str, path: Path) -> dict:
    def mean_pc(metodo):
        return np.mean([r["f1_per_class"] for r in rows if r["metodo"] == metodo and r["orcamento"] == 5], axis=0)

    f_floor, f_meth = mean_pc("piso"), mean_pc(method_label)
    gain = f_meth - f_floor
    order = np.argsort(gain)
    y = np.arange(NUM_CLASSES)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax1.barh(y + 0.2, f_floor[order], 0.38, color=COLORS["floor"], label="piso")
    ax1.barh(y - 0.2, f_meth[order], 0.38, color=COLORS["student"], label="aluno")
    ax1.set_yticks(y, [CLASSES[i] for i in order], fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("F1 no test_public (média de 3 seeds)")
    ax1.set_title("F1 por classe, 5 rótulos/classe")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", visible=False)
    ax2.barh(y, gain[order], 0.6, color=[COLORS["student"] if g > 0.02 else INK_MUTED for g in gain[order]])
    ax2.axvline(0, color=INK_MUTED, lw=1)
    ax2.set_yticks(y, [CLASSES[i] for i in order], fontsize=9)
    ax2.set_xlabel("F1 aluno − F1 piso")
    ax2.set_title("Ganho por classe (cinza: |ganho| ≤ 0,02)")
    ax2.grid(axis="y", visible=False)
    fig.suptitle("Parte 2 — onde os dados sem rótulo ajudaram (orçamento 5)", fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
    return dict(zip(CLASSES, gain.round(4).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", help="variante do aluno (padrão: a escolhida no val na Parte 1)")
    args = ap.parse_args()
    method = args.method or json.loads((OUTPUTS_DIR / "part1" / "student_results.json").read_text())["chosen"]
    print(f"método da Trilha C: aluno {method}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = run_all(method) + teacher_rows()
    method_label = f"aluno ({method})"
    # na raiz (entregável) e em outputs/part2 (no Colab, só outputs/ persiste no Drive)
    for csv_path in (ROOT / "curva_orcamento.csv", OUT / "curva_orcamento.csv"):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metodo", "orcamento", "seed", "f1_macro", "acuracia"])
            for r in rows:
                w.writerow([r["metodo"], r["orcamento"], r["seed"], f"{r['f1_macro']:.4f}", f"{r['acuracia']:.4f}"])

    summary = {m: summarize(rows, m) for m in ("piso", method_label, "professor zero-shot", "professor melhorado")}
    plot_curve(summary, method_label, OUT / "01_curva_orcamento.png")
    gain = plot_gain(rows, method_label, OUT / "02_ganho_por_classe_b5.png")
    (OUT / "results.json").write_text(json.dumps({"method": method, "summary": {k: {str(b): v for b, v in s.items()} for k, s in summary.items()},
                                                  "gain_per_class_b5": gain, "rows": rows}, indent=2))

    print("\n| método | 1/classe | 5/classe (3 seeds) | 20/classe |\n|---|---|---|---|")
    for m in ("piso", method_label, "professor melhorado", "professor zero-shot"):
        s = summary[m]
        print(f"| {m} | {s[1]['f1_mean']:.3f} | {s[5]['f1_mean']:.3f}" + (f" ± {s[5]['f1_std']:.3f}" if s[5]["n_seeds"] > 1 else "") + f" | {s[20]['f1_mean']:.3f} |")
    print(f"| teto | | {CEILING_F1} | |")
    print("\nganho por classe (b5):", dict(sorted(gain.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
