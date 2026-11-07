"""Parte 1, Trilha C — o aluno (ResNet-18) treinado nos pseudo-rótulos.

Treina as 3 variantes de pseudo-rótulo (ver pa3/student.py) no orçamento de 5
por classe, seed 0, escolhe a melhor PELO VAL, e compara com o professor e com
o piso: em quais classes o aluno supera o professor?

Retomável: variantes já treinadas (json em outputs/part1/students/) são lidas
em vez de retreinadas. Pressupõe `part0_floor.py` e `part1_teacher.py` já rodados.

Saídas em outputs/part1/:
  07_alunos_variantes.png         F1 val / test das 3 variantes + referências
  08_aluno_vs_professor_classe.png ganho por classe do aluno sobre o professor
  09_noite_vs_dia_todos.png        piso, professor e aluno, dia vs. noite
  10_matriz_confusao_aluno.png     matriz de confusão do aluno escolhido
  student_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pa3.config import CEILING_F1, CLASSES, OUTPUTS_DIR
from pa3.data import eval_dataset, is_night
from pa3.metrics import classification_metrics, confusion
from pa3.plotting import COLORS, INK_MUTED, plot_confusion, plt
from pa3.student import VARIANTS, train_student

OUT = OUTPUTS_DIR / "part1"
STUDENTS = OUT / "students"
LABELS = {
    "hard_zs": "duro · zero-shot\n(sem tratamento)",
    "hard_improved": "duro · professor\nmelhorado",
    "soft_improved": "suave · professor\nmelhorado",
}


def get_student(variant: str) -> dict:
    p = STUDENTS / f"{variant}_b5_s0.json"
    if p.exists():
        print(f"{variant}: já treinado, lendo {p.name}")
        return json.loads(p.read_text())
    return train_student(variant, budget=5, seed=0)


def by_light(y, pred, night):
    return {g: classification_metrics(y[m], pred[m])["f1_macro"] for g, m in (("dia", ~night), ("noite", night))}


def main():
    students = {v: get_student(v) for v in VARIANTS}
    chosen = max(students, key=lambda v: students[v]["best_val_f1"])
    print(f"\naluno escolhido pelo val: {chosen}")

    teacher = json.loads((OUT / "teacher_results.json").read_text())
    floor = json.loads((OUTPUTS_DIR / "part0" / "results.json").read_text())["resnet18"]
    t_zs, t_imp = teacher["teacher_zs"], teacher["teacher_improved_b5"]

    # --- 07: as três variantes -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = np.arange(len(VARIANTS))
    for k, (key, lbl, alpha) in enumerate((("best_val_f1", "F1 val (seleção)", 0.5), ("test_f1_macro", "F1 test_public", 1.0))):
        vals = [students[v][key] for v in VARIANTS]
        bars = ax.bar(xs + (k - 0.5) * 0.36, vals, 0.34, color=COLORS["student"], alpha=alpha, label=lbl)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    refs = [
        (floor["test_f1_macro"], f"piso (b) {floor['test_f1_macro']:.3f}", COLORS["floor"]),
        (t_zs["f1_macro"], f"professor zero-shot {t_zs['f1_macro']:.3f}", COLORS["teacher"]),
        (t_imp["f1_macro"], f"professor melhorado {t_imp['f1_macro']:.3f}", COLORS["teacher"]),
        (CEILING_F1, f"teto {CEILING_F1}", COLORS["ceiling"]),
    ]
    for y, lbl, c in refs:
        ax.axhline(y, color=c, ls="--" if "melhorado" not in lbl else ":", lw=1.3)
        ax.text(len(VARIANTS) - 0.45, y, " " + lbl, color=c, fontsize=8.5, va="center")
    ax.set_xticks(xs, [LABELS[v] + ("  ✓" if v == chosen else "") for v in VARIANTS])
    ax.set_xlim(-0.6, len(VARIANTS) + 0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 macro")
    ax.set_title("Aluno ResNet-18 — 3 formas de usar os pseudo-rótulos (5 rótulos/classe)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "07_alunos_variantes.png")
    plt.close(fig)

    # --- 08: aluno vs professor por classe ------------------------------------
    s = students[chosen]
    teacher_used = t_imp if VARIANTS[chosen]["teacher"] == "improved" else t_zs
    f_s, f_t, f_f = (np.array(x) for x in (s["test_f1_per_class"], teacher_used["f1_per_class"], floor["test_f1_per_class"]))
    order = np.argsort(f_s - f_t)
    y = np.arange(len(CLASSES))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.4, 1]})
    for k, (vals, lbl, c) in enumerate(((f_f, "piso (b)", COLORS["floor"]), (f_t, "professor", COLORS["teacher"]), (f_s, "aluno", COLORS["student"]))):
        ax1.barh(y + (k - 1) * 0.27, vals[order], 0.25, color=c, label=lbl)
    ax1.set_yticks(y, [CLASSES[i] for i in order], fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("F1 no test_public")
    ax1.set_title("F1 por classe")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", visible=False)
    d = (f_s - f_t)[order]
    ax2.barh(y, d, 0.6, color=[COLORS["student"] if v > 0 else COLORS["teacher"] for v in d])
    ax2.axvline(0, color=INK_MUTED, lw=1)
    ax2.set_yticks(y, [CLASSES[i] for i in order], fontsize=9)
    ax2.set_xlabel("F1 aluno − F1 professor")
    ax2.set_title("Onde o aluno supera o professor (laranja)")
    ax2.grid(axis="y", visible=False)
    fig.suptitle(f"Aluno ({chosen}) vs. professor que gerou seus rótulos — test_public", fontweight="bold")
    fig.savefig(OUT / "08_aluno_vs_professor_classe.png")
    plt.close(fig)

    # --- 09/10: dia vs noite + matriz do aluno --------------------------------
    ds = eval_dataset("test_public")
    y_test, night = np.asarray(ds.labels), is_night(ds.images)
    pred_s = np.load(STUDENTS / f"{chosen}_b5_s0_test_logits.npy").argmax(1)
    light = {
        "piso (b)": by_light(y_test, _floor_pred(ds), night),
        "professor": teacher_used["by_light"]["dia"]["f1_macro"], "_t_noite": teacher_used["by_light"]["noite"]["f1_macro"],
        "aluno": by_light(y_test, pred_s, night),
    }
    series = [
        ("piso (b)", [light["piso (b)"]["dia"], light["piso (b)"]["noite"]], COLORS["floor"]),
        ("professor", [light["professor"], light["_t_noite"]], COLORS["teacher"]),
        ("aluno", [light["aluno"]["dia"], light["aluno"]["noite"]], COLORS["student"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, (lbl, vals, c) in enumerate(series):
        bars = ax.bar(np.arange(2) + (k - 1) * 0.27, vals, 0.25, color=c, label=lbl)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.set_xticks(range(2), [f"dia (n={int((~night).sum())})", f"noite / IR (n={int(night.sum())})"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 macro no test_public")
    ax.set_title("Dia vs. noite: piso, professor e aluno")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "09_noite_vs_dia_todos.png")
    plt.close(fig)

    cm = confusion(y_test, pred_s)
    plot_confusion(cm, f"Aluno ResNet-18 ({chosen}), 5 rótulos/classe — test_public\nF1 macro {s['test_f1_macro']:.3f} · acurácia {s['test_accuracy']:.3f}",
                   OUT / "10_matriz_confusao_aluno.png")

    summary = {
        "chosen": chosen,
        "variants": {v: {k: students[v][k] for k in ("best_epoch", "best_val_f1", "test_f1_macro", "test_accuracy")} for v in VARIANTS},
        "per_class_student_minus_teacher": dict(zip(CLASSES, (f_s - f_t).tolist())),
        "per_class_student_minus_floor": dict(zip(CLASSES, (f_s - f_f).tolist())),
        "by_light": {"piso": light["piso (b)"], "professor": {"dia": light["professor"], "noite": light["_t_noite"]}, "aluno": light["aluno"]},
    }
    (OUT / "student_results.json").write_text(json.dumps(summary, indent=2))

    print("\n| variante | melhor época | F1 val | F1 test | acurácia test |\n|---|---|---|---|---|")
    for v in VARIANTS:
        r = students[v]
        print(f"| {v}{' ✓' if v == chosen else ''} | {r['best_epoch']} | {r['best_val_f1']:.3f} | {r['test_f1_macro']:.3f} | {r['test_accuracy']:.3f} |")
    print(f"\nreferências: piso {floor['test_f1_macro']:.3f} | professor zs {t_zs['f1_macro']:.3f} | professor melhorado {t_imp['f1_macro']:.3f} | teto {CEILING_F1}")
    print("aluno - professor por classe:", {c: round(v, 3) for c, v in sorted(summary["per_class_student_minus_teacher"].items(), key=lambda kv: -kv[1])})


def _floor_pred(ds):
    import torch

    from pa3.config import CKPT_DIR
    from pa3.engine import predict_logits
    from pa3.models import build_resnet18

    m = build_resnet18(pretrained=False)
    m.load_state_dict(torch.load(CKPT_DIR / "floor" / "resnet18_b5_s0.pt"))
    return predict_logits(m, ds).argmax(1)


if __name__ == "__main__":
    main()
