"""Parte 1, Trilha C — o aluno (ResNet-18) treinado nos pseudo-rótulos.

Treina as 6 variantes de pseudo-rótulo (ver pa3/student.py) no orçamento de 5
por classe, seed 0, escolhe a melhor PELO VAL, e compara com o professor e com
o piso: em quais classes o aluno supera o professor?

Retomável: variantes já treinadas (json em outputs/part1/students/) são lidas
em vez de retreinadas, e uma variante interrompida continua da última época.
Pressupõe os resultados de `part0_floor.py` e `part1_teacher.py` em outputs/
(versionados no git, então roda direto num clone, ex.: no Colab).

Saídas em outputs/part1/:
  07_alunos_variantes.png          F1 val / test das 6 variantes + referências
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
from pa3.metrics import confusion, subset_metrics
from pa3.plotting import COLORS, INK_MUTED, plot_confusion, plt
from pa3.student import VARIANTS, train_student

OUT = OUTPUTS_DIR / "part1"
STUDENTS = OUT / "students"
LABELS = {
    "hard_zs": "duro\nzero-shot\n(sem tratamento)",
    "hard_improved": "duro\nprof. melhorado",
    "soft_improved": "suave\nprof. melhorado",
    "soft_improved_tau30": "suave + filtro\nτ=0,3 (87%)",
    "soft_improved_tau50": "suave + filtro\nτ=0,5 (69%)",
    "soft_improved_tau70": "suave + filtro\nτ=0,7 (58%)",
}


def get_student(variant: str) -> dict:
    p = STUDENTS / f"{variant}_b5_s0.json"
    if p.exists():
        print(f"{variant}: já treinado, lendo {p.name}")
        return json.loads(p.read_text())
    return train_student(variant, budget=5, seed=0)


def by_light(y, pred, night):
    return {g: subset_metrics(y[m], pred[m]) for g, m in (("dia", ~night), ("noite", night))}


def floor_pred():
    """Previsões do piso (b) no test_public, a partir dos logits versionados em
    outputs/ (não precisa do checkpoint, que fica fora do git)."""
    return np.load(OUTPUTS_DIR / "floor_runs" / "resnet18_b5_s0_test_logits.npy").argmax(1)


def main():
    students = {v: get_student(v) for v in VARIANTS}
    chosen = max(students, key=lambda v: students[v]["best_val_f1"])
    print(f"\naluno escolhido pelo val: {chosen}")

    teacher = json.loads((OUT / "teacher_results.json").read_text())
    floor = json.loads((OUTPUTS_DIR / "part0" / "results.json").read_text())["resnet18"]
    t_zs, t_imp = teacher["teacher_zs"], teacher["teacher_improved_b5"]

    # --- 07: as variantes ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 5.6))
    xs = np.arange(len(VARIANTS))
    for k, (key, lbl, alpha) in enumerate((("best_val_f1", "F1 val (seleção)", 0.5), ("test_f1_macro", "F1 test_public", 1.0))):
        vals = [students[v][key] for v in VARIANTS]
        bars = ax.bar(xs + (k - 0.5) * 0.36, vals, 0.34, color=COLORS["student"], alpha=alpha, label=lbl)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    refs = [
        (floor["test_f1_macro"], f"piso (b) {floor['test_f1_macro']:.3f}", COLORS["floor"], "--"),
        (t_zs["f1_macro"], f"professor zero-shot {t_zs['f1_macro']:.3f}", COLORS["teacher"], "--"),
        (t_imp["f1_macro"], f"professor melhorado {t_imp['f1_macro']:.3f}", COLORS["teacher"], ":"),
        (CEILING_F1, f"teto {CEILING_F1}", COLORS["ceiling"], "--"),
    ]
    for y, lbl, c, ls in refs:
        ax.axhline(y, color=c, ls=ls, lw=1.3, label=lbl)  # na legenda: teto e professor melhorado quase coincidem
    ax.set_xticks(xs, [LABELS[v] + ("\n✓ escolhido no val" if v == chosen else "") for v in VARIANTS], fontsize=9)
    ax.set_xlim(-0.6, len(VARIANTS) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 macro")
    ax.set_title("Aluno ResNet-18 — formas de usar os pseudo-rótulos (5 rótulos/classe)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "07_alunos_variantes.png")
    plt.close(fig)

    # --- 08: aluno vs professor por classe ------------------------------------
    s = students[chosen]
    teacher_used = t_imp if VARIANTS[chosen]["teacher"] == "improved" else t_zs
    f_s, f_t, f_f = (np.array(x) for x in (s["test_f1_per_class"], teacher_used["f1_per_class"], floor["test_f1_per_class"]))
    order = np.argsort(f_s - f_t)
    y = np.arange(len(CLASSES))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.38})
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
        "piso (b)": by_light(y_test, floor_pred(), night),
        "professor": teacher_used["by_light"],
        "aluno": by_light(y_test, pred_s, night),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, metric, name in zip(axes, ("accuracy", "f1_macro"), ("acurácia", "F1 macro (classes presentes)")):
        for k, (lbl, c) in enumerate((("piso (b)", COLORS["floor"]), ("professor", COLORS["teacher"]), ("aluno", COLORS["student"]))):
            vals = [light[lbl][g][metric] for g in ("dia", "noite")]
            bars = ax.bar(np.arange(2) + (k - 1) * 0.27, vals, 0.25, color=c, label=lbl)
            ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
        ax.set_xticks(range(2), [f"dia (n={int((~night).sum())})", f"noite / IR (n={int(night.sum())})"])
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{name} no test_public")
        ax.set_title(name[0].upper() + name[1:])
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    fig.suptitle("Dia vs. noite: piso, professor e aluno", fontweight="bold")
    fig.savefig(OUT / "09_noite_vs_dia_todos.png")
    plt.close(fig)

    cm = confusion(y_test, pred_s)
    plot_confusion(cm, f"Aluno ResNet-18 ({chosen}), 5 rótulos/classe — test_public\nF1 macro {s['test_f1_macro']:.3f} · acurácia {s['test_accuracy']:.3f}",
                   OUT / "10_matriz_confusao_aluno.png")

    summary = {
        "chosen": chosen,
        "variants": {v: {k: students[v].get(k) for k in ("n_train", "best_epoch", "best_val_f1", "test_f1_macro", "test_accuracy")} for v in VARIANTS},
        "per_class_student_minus_teacher": dict(zip(CLASSES, (f_s - f_t).tolist())),
        "per_class_student_minus_floor": dict(zip(CLASSES, (f_s - f_f).tolist())),
        "by_light": light,
    }
    (OUT / "student_results.json").write_text(json.dumps(summary, indent=2))

    print("\n| variante | imagens de treino | melhor época | F1 val | F1 test | acurácia test |\n|---|---|---|---|---|---|")
    for v in VARIANTS:
        r = students[v]
        print(f"| {v}{' (escolhido)' if v == chosen else ''} | {r.get('n_train', '-')} | {r['best_epoch']} | {r['best_val_f1']:.3f} | {r['test_f1_macro']:.3f} | {r['test_accuracy']:.3f} |")
    print(f"\nreferências: piso {floor['test_f1_macro']:.3f} | professor zs {t_zs['f1_macro']:.3f} | professor melhorado {t_imp['f1_macro']:.3f} | teto {CEILING_F1}")
    print("aluno - professor por classe:", {c: round(v, 3) for c, v in sorted(summary["per_class_student_minus_teacher"].items(), key=lambda kv: -kv[1])})


if __name__ == "__main__":
    main()
