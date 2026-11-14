"""Parte 1 — o professor multi-crop: o BioCLIP 2 vendo a foto em 3 recortes.

O professor original usa center crop (venceu o letterbox nos 105 rótulos) e
por isso perde o animal quando ele está na lateral da foto (figura 11). Aqui
o mesmo BioCLIP 2, com os mesmos protótipos do professor melhorado de cada
orçamento (beta e temperatura de part1_teacher.py), é aplicado ao quadrado
esquerdo, central e direito da foto, e as 3 vistas são combinadas de uma de
duas formas:

  media        média das probabilidades das 3 vistas
  max_conf     a vista em que o professor está mais confiante

A forma de combinar é escolhida SÓ nos 105 rótulos do orçamento 5 (com
leave-one-out para o professor melhorado, como na escolha de beta) e aplicada
igual nos orçamentos 1 e 20. O test_public é só reportado.

Gera os pseudo-rótulos `improved_mc` (orçamentos 1, 5 e 20) para os alunos
de pa3/student.py. Precisa dos embeddings das 3 vistas no train inteiro: a
vista central e as do test_public vêm versionadas
(outputs/part1/teacher_embeddings/); as laterais do train são calculadas na
1ª vez (~10 min numa GPU T4; horas em CPU).

Saídas em outputs/part1/:
  teacher_multicrop.json                     seleção, resultados no teste, por orçamento
  teacher_test_probs_improved_mc_b5.npy      probabilidades no test_public (ordem: ids ordenados)
  03_matriz_ruido_teacher_improved_mc_b5.png matriz de ruído desse professor
  pseudo_labels/pseudo_improved_mc_b{1,5,20}.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from part1_teacher import noise_analysis

from pa3.config import BUDGETS, CLASSES, OUTPUTS_DIR
from pa3.data import budget_labels, get_cache, is_night, split_labels
from pa3.metrics import classification_metrics
from pa3.student import pseudo_label_path
from pa3.teacher import (LOGIT_SCALE, VERSIONED_EMB_DIR, cached_embeddings, embedding_path, load_teacher,
                         prototype_classifier, softmax)

OUT = OUTPUTS_DIR / "part1"
VIEWS = ("center", "left", "right")
HOWS = ("center", "media", "max_conf")
MD_CONF, CROP_MARGIN = 0.2, 0.125  # os mesmos de part1_size_analysis.py


def combine(probs: dict[str, np.ndarray], how: str) -> np.ndarray:
    if how == "center":
        return probs["center"]
    stack = np.stack([probs[v] for v in VIEWS])  # (3, N, C)
    if how == "media":
        return stack.mean(0)
    best = stack.max(2).argmax(0)  # max_conf: a vista mais confiante de cada imagem
    return stack[best, np.arange(stack.shape[1])]


def f1(y, p) -> float:
    return classification_metrics(y, p.argmax(1))["f1_macro"]


def view_probs(emb: dict[str, np.ndarray], rows, protos: np.ndarray, T: float) -> dict[str, np.ndarray]:
    return {v: softmax(LOGIT_SCALE * emb[v][rows] @ protos.T, T) for v in VIEWS}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr = json.loads((OUT / "teacher_results.json").read_text())
    assert tr["teacher"] == "bioclip2" and tr["prompt"] == "ensemble"
    text = np.load(OUT / "_text_bioclip2.npy")[1]  # prompts "ensemble"

    # --- embeddings das 3 vistas: train inteiro e test_public ---------------
    def have(view, split):
        p = embedding_path("bioclip2", view, split)
        return p.exists() or (VERSIONED_EMB_DIR / p.name).exists()

    missing = [(v, s) for v in VIEWS for s in ("train", "test_public") if not have(v, s)]
    print(f"embeddings a calcular: {missing or 'nenhum'}", flush=True)
    bundle = load_teacher("bioclip2") if missing else None  # carrega o BioCLIP 2 uma vez só
    emb_tr, emb_te = {}, {}
    tr_ids = te_ids = None
    for v in VIEWS:
        ids_v, emb_tr[v] = cached_embeddings("bioclip2", v, "train", model_bundle=bundle)
        tr_ids = tr_ids or ids_v
        assert ids_v == tr_ids
        ids_t, emb_te[v] = cached_embeddings("bioclip2", v, "test_public", model_bundle=bundle)
        te_ids = te_ids or ids_t
        assert ids_t == te_ids
    tidx = {i: k for k, i in enumerate(tr_ids)}

    test = split_labels("test_public")
    assert te_ids == sorted(test)
    yt = np.array([test[i] for i in te_ids])

    # --- seleção da combinação nos 105 rótulos (orçamento 5, leave-one-out) --
    b5 = tr["budget_5"]
    bl5 = budget_labels(5)
    ids5 = sorted(bl5)
    y5 = np.array([bl5[i] for i in ids5])
    rows5 = [tidx[i] for i in ids5]
    loo = {v: np.zeros((len(ids5), len(CLASSES))) for v in VIEWS}
    for i in range(len(ids5)):
        keep = np.arange(len(ids5)) != i
        protos = prototype_classifier(text, emb_tr["center"][rows5][keep], y5[keep], b5["beta"])
        for v in VIEWS:
            loo[v][i] = softmax(LOGIT_SCALE * emb_tr[v][rows5[i] : rows5[i] + 1] @ protos.T, b5["T_improved"])[0]
    selection = {h: f1(y5, combine(loo, h)) for h in HOWS}
    chosen = max(HOWS, key=lambda h: (round(selection[h], 4), h == "center"))  # empate: fica o center
    print(f"seleção nos 105 rótulos (F1 leave-one-out): { {h: round(s, 3) for h, s in selection.items()} } -> {chosen}")

    # --- animal na borda do center crop (para o diagnóstico no teste) --------
    md = json.loads((OUT / "megadetector" / "test_public.json").read_text())["detections"]
    edge = np.zeros(len(te_ids), dtype=bool)
    for k, iid in enumerate(te_ids):
        b = [d[2:] for d in md.get(iid, []) if d[0] == "animal" and d[1] >= MD_CONF]
        if b:
            x1, _, x2, _ = max(b, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            inside = max(0.0, min(x2, 1 - CROP_MARGIN) - max(x1, CROP_MARGIN))
            edge[k] = x2 > x1 and 1 - inside / (x2 - x1) >= 0.5

    # --- professor multi-crop em cada orçamento: pseudo-rótulos + teste ------
    res = {"views": VIEWS, "selection_105_loo": selection, "chosen": chosen, "n_edge": int(edge.sum()), "budgets": {}}
    all_rows = np.arange(len(tr_ids))
    for budget in BUDGETS:
        p = tr[f"budget_{budget}"]
        bl = budget_labels(budget)
        ids_b = sorted(bl)
        protos = prototype_classifier(text, emb_tr["center"][[tidx[i] for i in ids_b]], np.array([bl[i] for i in ids_b]), p["beta"])
        probs_tr = combine(view_probs(emb_tr, all_rows, protos, p["T_improved"]), chosen)
        np.savez(pseudo_label_path("improved_mc", budget), ids=np.array(tr_ids), probs=probs_tr.astype(np.float32))

        te = view_probs(emb_te, np.arange(len(te_ids)), protos, p["T_improved"])
        # sanidade: só com a vista central, tem que reproduzir o professor melhorado já reportado
        assert abs(f1(yt, te["center"]) - p["improved_test_f1_macro"]) < 1e-6, f"orçamento {budget}: vista central não reproduz"
        out_b = {}
        for h in HOWS:
            pred = combine(te, h).argmax(1)
            m = classification_metrics(yt, pred)
            out_b[h] = {"f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
                        "accuracy_animal_na_borda": float((pred[edge] == yt[edge]).mean()),
                        "accuracy_animal_no_centro": float((pred[~edge] == yt[~edge]).mean())}
        res["budgets"][str(budget)] = {"beta": p["beta"], "T_improved": p["T_improved"], "test_public": out_b,
                                       "pseudo_kept_tau70": float((probs_tr.max(1) >= 0.7).mean())}
        print(f"orçamento {budget}: " + " | ".join(f"{h} F1 {r['f1_macro']:.3f} (borda {r['accuracy_animal_na_borda']:.2f})"
                                                for h, r in out_b.items()))
        if budget == 5:
            probs_te5 = combine(te, chosen)

    # --- o professor multi-crop do orçamento 5, no formato de teacher_results
    np.save(OUT / "teacher_test_probs_improved_mc_b5.npy", probs_te5)
    night = is_night(get_cache("test_public").subset(te_ids))
    noise_analysis("BioCLIP 2", yt, probs_te5, night, res, "teacher_improved_mc_b5",
                   "Matriz de ruído: professor BioCLIP 2 melhorado, 3 recortes, 5 rótulos/classe — test_public")
    (OUT / "teacher_multicrop.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    r = res["teacher_improved_mc_b5"]
    print(f"professor multi-crop (orçamento 5, {chosen}): F1 {r['f1_macro']:.3f} | acurácia {r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
