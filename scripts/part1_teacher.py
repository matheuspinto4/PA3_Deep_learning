"""Parte 1, Trilha C — o professor.

1. Escolha do professor (modelo x vista da imagem x prompts) usando SÓ os 105
   rótulos do orçamento (que são do train). O val não é tocado.
2. F1 zero-shot de cada professor no test_public com os nomes das classes
   como prompt (linha de referência obrigatória) e com o ensemble de prompts.
3. Professor melhorado com os rótulos do orçamento (protótipos texto+imagem,
   beta e temperatura escolhidos por leave-one-out nos próprios rótulos).
4. Matriz de ruído (confusão do professor no test_public) e onde ele erra de
   forma sistemática: noite vs. dia, pares de classes, viés de distribuição.
5. Pseudo-rótulos do train para os orçamentos 1, 5 e 20 (usados pelo aluno).

Saídas em outputs/part1/ (figuras 01-07 + teacher_results.json) e
cache/teacher/pseudo_*.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pa3.config import BUDGETS, CEILING_F1, CLASSES, OUTPUTS_DIR
from pa3.data import budget_labels, get_cache, is_night, split_labels
from pa3.metrics import classification_metrics, confusion
from pa3.plotting import COLORS, INK_MUTED, plot_confusion, plt
from pa3.student import pseudo_label_path
from pa3.teacher import (
    TEACHERS,
    cached_embeddings,
    fit_temperature,
    load_teacher,
    loo_prototype_predictions,
    prototype_classifier,
    softmax,
    text_classifier,
    zero_shot_logits,
)

OUT = OUTPUTS_DIR / "part1"
OUT.mkdir(parents=True, exist_ok=True)
VIEWS = ("center", "letterbox")
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_BETA = 1.0  # orçamento 1: sem leave-one-out possível (1 imagem por classe)


def labeled_arrays(labels: dict[str, int]):
    ids = sorted(labels)
    return ids, np.array([labels[i] for i in ids])


def f1(y, logits):
    return classification_metrics(y, logits.argmax(1))["f1_macro"]


# ----------------------------------------------------------------------------
# 1-2. Escolha do professor e zero-shot no test_public
# ----------------------------------------------------------------------------


def select_teacher(results: dict) -> tuple[str, str, str]:
    ids5, y5 = labeled_arrays(budget_labels(5))
    test_ids, y_test = labeled_arrays(split_labels("test_public"))
    sel, zs_test = {}, {}
    for name in TEACHERS:
        print(f"\n== professor {name} ==", flush=True)
        bundle = load_teacher(name)
        model, _, tok = bundle
        texts = {p: text_classifier(model, tok, p) for p in ("basic", "ensemble")}
        for view in VIEWS:
            _, emb = cached_embeddings(name, view, "train", ids5, model_bundle=bundle)
            for p, t in texts.items():
                sel[f"{name}|{view}|{p}"] = f1(y5, zero_shot_logits(emb, t))
                print(f"  105 rótulos  {view:9s} {p:8s} F1 = {sel[f'{name}|{view}|{p}']:.3f}", flush=True)
        best_view = max(VIEWS, key=lambda v: max(sel[f"{name}|{v}|{p}"] for p in texts))
        _, emb_test = cached_embeddings(name, best_view, "test_public", test_ids, model_bundle=bundle)
        for p, t in texts.items():
            m = classification_metrics(y_test, zero_shot_logits(emb_test, t).argmax(1))
            zs_test[f"{name}|{best_view}|{p}"] = {"f1_macro": m["f1_macro"], "accuracy": m["accuracy"]}
            print(f"  test_public {best_view:9s} {p:8s} F1 = {m['f1_macro']:.3f} acc = {m['accuracy']:.3f}", flush=True)
        np.save(OUT / f"_text_{name}.npy", np.stack([texts["basic"], texts["ensemble"]]))
        del model, bundle

    best = max(sel, key=sel.get)
    results["selection_105"] = sel
    results["zero_shot_test_public"] = zs_test
    results["chosen"] = best
    print(f"\nprofessor escolhido (F1 nos 105 rótulos): {best}")
    return tuple(best.split("|"))


def plot_selection(results: dict, path: Path) -> None:
    sel, zs = results["selection_105"], results["zero_shot_test_public"]
    names = list(TEACHERS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    configs = [(v, p) for v in VIEWS for p in ("basic", "ensemble")]
    shades = {("center", "basic"): 0.35, ("center", "ensemble"): 0.6, ("letterbox", "basic"): 0.8, ("letterbox", "ensemble"): 1.0}
    w = 0.2
    for k, (v, p) in enumerate(configs):
        vals = [sel[f"{n}|{v}|{p}"] for n in names]
        bars = ax1.bar(np.arange(len(names)) + (k - 1.5) * w, vals, w * 0.92, color=COLORS["teacher"], alpha=shades[(v, p)], label=f"{v} · {p}")
        ax1.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax1.set_xticks(range(len(names)), names)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("F1 macro")
    ax1.set_title("Escolha do professor: zero-shot nos 105 rótulos do train")
    ax1.legend(title="vista · prompt", fontsize=9, loc="upper left")
    ax1.grid(axis="x", visible=False)

    w = 0.36
    for k, p in enumerate(("basic", "ensemble")):
        keys = [next(key for key in zs if key.startswith(n + "|") and key.endswith("|" + p)) for n in names]
        vals = [zs[key]["f1_macro"] for key in keys]
        bars = ax2.bar(np.arange(len(names)) + (k - 0.5) * w, vals, w * 0.92, color=COLORS["teacher"], alpha=0.5 if p == "basic" else 1.0,
                       label="nomes das classes (basic)" if p == "basic" else "ensemble de prompts")
        ax2.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    floor = _floor_f1()
    if floor is not None:
        ax2.axhline(floor, color=COLORS["floor"], ls="--", lw=1.5)
        ax2.text(len(names) - 0.5, floor + 0.01, f"piso (b) = {floor:.3f}", color=COLORS["floor"], ha="right", fontsize=9)
    ax2.axhline(CEILING_F1, color=COLORS["ceiling"], ls="--", lw=1.5)
    ax2.text(len(names) - 0.5, CEILING_F1 + 0.01, f"teto = {CEILING_F1}", color=COLORS["ceiling"], ha="right", fontsize=9)
    ax2.set_xticks(range(len(names)), [f"{n}\n({next(k for k in zs if k.startswith(n + '|')).split('|')[1]})" for n in names])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("F1 macro")
    ax2.set_title("F1 zero-shot no test_public (referência)")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def _floor_f1():
    p = OUTPUTS_DIR / "part0" / "results.json"
    if p.exists():
        return json.loads(p.read_text())["resnet18"]["test_f1_macro"]
    return None


# ----------------------------------------------------------------------------
# 3. Professor melhorado + pseudo-rótulos por orçamento
# ----------------------------------------------------------------------------


def teacher_for_budget(budget: int, text: np.ndarray, emb_of: dict, train_index: dict) -> dict:
    """Ajusta zero-shot (só temperatura) e melhorado (beta + temperatura) com os
    rótulos do orçamento e devolve os dois classificadores."""
    ids, y = labeled_arrays(budget_labels(budget))
    emb = emb_of["train"][[train_index[i] for i in ids]]

    zs_logits = zero_shot_logits(emb, text)
    T_zs = fit_temperature(zs_logits, y)

    if budget > 1:
        loo = {b: loo_prototype_predictions(text, emb, y, b) for b in BETAS}
        loo_f1 = {b: f1(y, l) for b, l in loo.items()}
        # empate: o beta menor (mais perto do texto, menos sobreajuste às poucas imagens)
        beta = max(BETAS, key=lambda b: (round(loo_f1[b], 4), -b))
        T_imp = fit_temperature(loo[beta], y)
    else:
        loo_f1, beta = {}, DEFAULT_BETA
        T_imp = T_zs
    protos = prototype_classifier(text, emb, y, beta)
    return {"budget": budget, "T_zs": T_zs, "beta": beta, "loo_f1": {str(k): v for k, v in loo_f1.items()}, "T_improved": T_imp, "protos": protos}


def make_pseudo_labels(tb: dict, text: np.ndarray, emb_train: np.ndarray, train_ids: list[str]) -> None:
    for variant, logits, T in (
        ("zs", zero_shot_logits(emb_train, text), tb["T_zs"]),
        ("improved", zero_shot_logits(emb_train, tb["protos"]), tb["T_improved"]),
    ):
        np.savez(pseudo_label_path(variant, tb["budget"]), ids=np.array(train_ids), probs=softmax(logits, T).astype(np.float32))


# ----------------------------------------------------------------------------
# 4. Análise do ruído
# ----------------------------------------------------------------------------


def noise_analysis(name: str, y_test: np.ndarray, probs: np.ndarray, night: np.ndarray, results: dict, tag: str, title: str) -> None:
    pred = probs.argmax(1)
    cm = confusion(y_test, pred)
    m = classification_metrics(y_test, pred)
    plot_confusion(cm, f"{title}\nF1 macro {m['f1_macro']:.3f} · acurácia {m['accuracy']:.3f}", OUT / f"03_matriz_ruido_{tag}.png")
    by_light = {}
    for lbl, mask in (("dia", ~night), ("noite", night)):
        mm = classification_metrics(y_test[mask], pred[mask])
        by_light[lbl] = {"n": int(mask.sum()), "accuracy": mm["accuracy"], "f1_macro": mm["f1_macro"]}
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [
        {"true": CLASSES[i], "pred": CLASSES[j], "count": int(off[i, j]), "frac_of_true": float(off[i, j] / cm[i].sum())}
        for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:12], off.shape))
    ]
    results[tag] = {
        "f1_macro": m["f1_macro"],
        "accuracy": m["accuracy"],
        "f1_per_class": m["f1_per_class"],
        "recall_per_class": (np.diag(cm) / cm.sum(1)).tolist(),
        "precision_per_class": (np.diag(cm) / np.maximum(cm.sum(0), 1)).tolist(),
        "predicted_counts": cm.sum(0).tolist(),
        "true_counts": cm.sum(1).tolist(),
        "by_light": by_light,
        "top_confusions": pairs,
        "confusion_matrix": cm.tolist(),
    }


def plot_light(results: dict, path: Path) -> None:
    tags = [("teacher_zs", "professor zero-shot"), ("teacher_improved_b5", "professor melhorado (5/classe)")]
    floor_light = _floor_by_light()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    groups = ["dia", "noite"]
    series = [(lbl, [results[t]["by_light"][g]["f1_macro"] for g in groups], COLORS["teacher"], a) for (t, lbl), a in zip(tags, (0.55, 1.0))]
    if floor_light:
        series.insert(0, ("piso (b)", [floor_light[g]["f1_macro"] for g in groups], COLORS["floor"], 1.0))
    w = 0.8 / len(series)
    for k, (lbl, vals, color, alpha) in enumerate(series):
        bars = ax.bar(np.arange(2) + (k - (len(series) - 1) / 2) * w, vals, w * 0.92, color=color, alpha=alpha, label=lbl)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    n = results["teacher_zs"]["by_light"]
    ax.set_xticks(range(2), [f"dia (n={n['dia']['n']})", f"noite / IR (n={n['noite']['n']})"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 macro no test_public")
    ax.set_title("O professor erra mais à noite?")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def _floor_by_light():
    p = OUTPUTS_DIR / "part0" / "results.json"
    if not p.exists():
        return None
    import torch

    from pa3.data import eval_dataset
    from pa3.engine import predict_logits
    from pa3.floor import CKPT_DIR
    from pa3.models import build_resnet18

    m = build_resnet18(pretrained=False)
    m.load_state_dict(torch.load(CKPT_DIR / "floor" / "resnet18_b5_s0.pt"))
    ds = eval_dataset("test_public")
    pred = predict_logits(m, ds).argmax(1)
    y = np.asarray(ds.labels)
    night = is_night(ds.images)
    return {g: {"f1_macro": classification_metrics(y[mask], pred[mask])["f1_macro"]} for g, mask in (("dia", ~night), ("noite", night))}


def plot_bias(results: dict, train_counts: dict, path: Path) -> None:
    """Viés de distribuição: quantas vezes o professor prevê cada classe vs.
    quantas existem (test_public), e a distribuição dos pseudo-rótulos no train."""
    r = results["teacher_zs"]
    true, pred = np.array(r["true_counts"]), np.array(r["predicted_counts"])
    order = np.argsort(true)[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    y = np.arange(len(CLASSES))
    ax1.barh(y + 0.2, true[order], 0.38, color=INK_MUTED, alpha=0.5, label="verdadeiro")
    ax1.barh(y - 0.2, pred[order], 0.38, color=COLORS["teacher"], label="previsto pelo professor")
    ax1.set_yticks(y, [CLASSES[i] for i in order], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("nº de imagens no test_public")
    ax1.set_title("Para onde vão as previsões do professor zero-shot")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", visible=False)

    zs, imp = np.array(train_counts["zs"]), np.array(train_counts["improved_b5"])
    order2 = np.argsort(zs)[::-1]
    ax2.barh(y + 0.2, zs[order2], 0.38, color=COLORS["teacher"], alpha=0.55, label="zero-shot")
    ax2.barh(y - 0.2, imp[order2], 0.38, color=COLORS["teacher"], label="melhorado (5/classe)")
    ax2.set_yticks(y, [CLASSES[i] for i in order2], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("nº de pseudo-rótulos no train (6.505 imagens)")
    ax2.set_title("Distribuição dos pseudo-rótulos no train")
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_precision_coverage(y_test: np.ndarray, probs_by_variant: dict, path: Path) -> dict:
    """Diagnóstico (não é seleção): se o professor rotulasse o test_public, que
    precisão e cobertura os pseudo-rótulos teriam com um filtro de confiança?"""
    taus = np.linspace(0, 0.95, 39)
    out = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for (lbl, probs), alpha in zip(probs_by_variant.items(), (0.55, 1.0)):
        conf, pred = probs.max(1), probs.argmax(1)
        prec = [float((pred[conf >= t] == y_test[conf >= t]).mean()) if (conf >= t).any() else np.nan for t in taus]
        cov = [float((conf >= t).mean()) for t in taus]
        ax.plot(cov, prec, color=COLORS["teacher"], alpha=alpha, label=lbl)
        for t in (0.5, 0.8):
            k = int(np.argmin(np.abs(taus - t)))
            ax.plot(cov[k], prec[k], "o", ms=8, color=COLORS["teacher"], alpha=alpha, mec="white", mew=2)
            ax.annotate(f"τ={t}", (cov[k], prec[k]), textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK_MUTED)
        out[lbl] = {"tau": taus.tolist(), "precision": prec, "coverage": cov}
    ax.set_xlabel("cobertura (fração das imagens com confiança ≥ τ)")
    ax.set_ylabel("precisão dos pseudo-rótulos")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Pseudo-rótulos do professor: precisão × cobertura (test_public)")
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------


def main():
    results = {}
    name, view, prompt = select_teacher(results)
    plot_selection(results, OUT / "01_escolha_professor.png")
    results["teacher"], results["view"], results["prompt"] = name, view, prompt

    print(f"\n== embeddings completos: {name} / {view} ==", flush=True)
    bundle = load_teacher(name)
    train_ids, emb_train = cached_embeddings(name, view, "train", model_bundle=bundle)
    test_all_ids, emb_test_all = cached_embeddings(name, view, "test_public", model_bundle=bundle)
    text = np.load(OUT / f"_text_{name}.npy")[0 if prompt == "basic" else 1]
    train_index = {i: k for k, i in enumerate(train_ids)}

    test_labels = split_labels("test_public")
    test_idx = [k for k, i in enumerate(test_all_ids) if i in test_labels]
    y_test = np.array([test_labels[test_all_ids[k]] for k in test_idx])
    emb_test = emb_test_all[test_idx]
    tc = get_cache("test_public")
    night = is_night(tc.subset([test_all_ids[k] for k in test_idx]))

    train_counts = {}
    for budget in BUDGETS:
        tb = teacher_for_budget(budget, text, {"train": emb_train}, train_index)
        make_pseudo_labels(tb, text, emb_train, train_ids)
        zs_test = softmax(zero_shot_logits(emb_test, text), tb["T_zs"])
        imp_test = softmax(zero_shot_logits(emb_test, tb["protos"]), tb["T_improved"])
        m_imp = classification_metrics(y_test, imp_test.argmax(1))
        results[f"budget_{budget}"] = {
            "T_zs": tb["T_zs"], "beta": tb["beta"], "loo_f1": tb["loo_f1"], "T_improved": tb["T_improved"],
            "improved_test_f1_macro": m_imp["f1_macro"], "improved_test_accuracy": m_imp["accuracy"],
            "improved_test_f1_per_class": m_imp["f1_per_class"],
        }
        print(f"orçamento {budget}: beta={tb['beta']} T_zs={tb['T_zs']:.2f} T_imp={tb['T_improved']:.2f} "
              f"| professor melhorado test F1 {m_imp['f1_macro']:.3f} acc {m_imp['accuracy']:.3f}", flush=True)
        if budget == 5:
            noise_analysis(name, y_test, zs_test, night, results, "teacher_zs", f"Matriz de ruído: professor {name} zero-shot ({prompt}) — test_public")
            noise_analysis(name, y_test, imp_test, night, results, "teacher_improved_b5", f"Matriz de ruído: professor {name} melhorado com 5 rótulos/classe — test_public")
            train_counts["zs"] = np.bincount(zero_shot_logits(emb_train, text).argmax(1), minlength=len(CLASSES)).tolist()
            train_counts["improved_b5"] = np.bincount(zero_shot_logits(emb_train, tb["protos"]).argmax(1), minlength=len(CLASSES)).tolist()
            results["precision_coverage"] = plot_precision_coverage(
                y_test, {"zero-shot": zs_test, "melhorado (5/classe)": imp_test}, OUT / "06_precisao_cobertura.png"
            )
            np.save(OUT / "teacher_test_probs_zs.npy", zs_test)
            np.save(OUT / "teacher_test_probs_improved_b5.npy", imp_test)
    results["train_pseudo_counts"] = train_counts
    plot_light(results, OUT / "04_professor_noite_vs_dia.png")
    plot_bias(results, train_counts, OUT / "05_vies_distribuicao.png")
    (OUT / "teacher_results.json").write_text(json.dumps(results, indent=2))

    r = results["teacher_zs"]
    print(f"\nprofessor zero-shot test_public: F1 {r['f1_macro']:.3f} acc {r['accuracy']:.3f} | dia/noite: {r['by_light']}")
    print("pares mais confundidos:")
    for p in r["top_confusions"][:8]:
        print(f"  {p['true']:>22s} -> {p['pred']:<22s} {p['count']:3d} ({p['frac_of_true']:.0%} da classe)")


if __name__ == "__main__":
    main()
