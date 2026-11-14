"""Parte 1 (análise) — e se o professor visse a foto inteira em 3 recortes?

O professor escolhido usa center crop (venceu o letterbox nos 105 rótulos),
e por isso perde o animal quando ele está na lateral da foto (figura 11).
Aqui testamos um professor multi-crop, sem treino: o mesmo BioCLIP 2, os
mesmos protótipos do professor melhorado (orçamento 5), aplicados ao
quadrado esquerdo, central e direito da foto, combinando as 3 vistas de
duas formas:

  media        média das probabilidades das 3 vistas
  max_conf     a vista em que o professor está mais confiante

A forma de combinar é escolhida SÓ nos 105 rótulos do orçamento (com
leave-one-out para o professor melhorado, como na escolha de beta). O
test_public é só reportado, incluindo a fatia "animal na borda" medida
com o MegaDetector. Os pseudo-rótulos e os alunos NÃO foram refeitos com esse
professor: é o próximo passo, registrado no README.

Saída: outputs/part1/teacher_multicrop.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pa3.config import CLASSES, OUTPUTS_DIR
from pa3.data import budget_labels, split_labels
from pa3.metrics import classification_metrics
from pa3.teacher import (LOGIT_SCALE, cached_embeddings, load_teacher, prototype_classifier, softmax,
                         zero_shot_logits)

OUT = OUTPUTS_DIR / "part1"
VIEWS = ("center", "left", "right")
MD_CONF, CROP_MARGIN = 0.2, 0.125  # os mesmos de part1_size_analysis.py


def combine(probs: dict[str, np.ndarray], how: str) -> np.ndarray:
    if how == "center":
        return probs["center"]
    stack = np.stack([probs[v] for v in VIEWS])  # (3, N, C)
    if how == "media":
        return stack.mean(0)
    best = stack.max(2).argmax(0)  # max_conf: vista mais confiante de cada imagem
    return stack[best, np.arange(stack.shape[1])]


def f1(y, p) -> float:
    return classification_metrics(y, p.argmax(1))["f1_macro"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr = json.loads((OUT / "teacher_results.json").read_text())
    assert tr["teacher"] == "bioclip2" and tr["prompt"] == "ensemble"
    b5 = tr["budget_5"]
    beta, T_zs, T_imp = b5["beta"], b5["T_zs"], b5["T_improved"]
    text = np.load(OUT / "_text_bioclip2.npy")[1]  # prompts "ensemble"

    bl = budget_labels(5)
    ids = sorted(bl)
    y = np.array([bl[i] for i in ids])
    test = split_labels("test_public")
    tids = sorted(test)
    yt = np.array([test[i] for i in tids])

    bundle = None
    emb_b, emb_t = {}, {}
    for v in VIEWS:
        need = v != "center"
        if need and bundle is None:
            bundle = load_teacher("bioclip2")
        _, emb_b[v] = cached_embeddings("bioclip2", v, "train", ids=ids, model_bundle=bundle)
        if v == "center":
            _, emb_t[v] = cached_embeddings("bioclip2", v, "test_public", ids=tids)
        else:
            _, emb_t[v] = cached_embeddings("bioclip2", v, "test_public", model_bundle=bundle)

    # --- seleção nos 105 rótulos ---------------------------------------------
    # zero-shot: nada é ajustado nas imagens, então basta aplicar
    zs_b = {v: softmax(zero_shot_logits(emb_b[v], text), T_zs) for v in VIEWS}
    # melhorado: protótipos montados SEM a imagem avaliada (leave-one-out),
    # sempre com os embeddings centrais das imagens rotuladas, como no professor original
    imp_b = {v: np.zeros((len(ids), len(CLASSES))) for v in VIEWS}
    for i in range(len(ids)):
        keep = np.arange(len(ids)) != i
        protos = prototype_classifier(text, emb_b["center"][keep], y[keep], beta)
        for v in VIEWS:
            imp_b[v][i] = softmax(LOGIT_SCALE * emb_b[v][i : i + 1] @ protos.T, T_imp)[0]
    hows = ("center", "media", "max_conf")
    sel = {"zero_shot": {h: f1(y, combine(zs_b, h)) for h in hows},
           "melhorado_loo": {h: f1(y, combine(imp_b, h)) for h in hows}}
    chosen = max(hows, key=lambda h: (round(sel["melhorado_loo"][h], 4), h == "center"))  # empate: fica o center

    # --- test_public (só reportado) --------------------------------------------
    protos = prototype_classifier(text, emb_b["center"], y, beta)
    zs_t = {v: softmax(zero_shot_logits(emb_t[v], text), T_zs) for v in VIEWS}
    imp_t = {v: softmax(LOGIT_SCALE * emb_t[v] @ protos.T, T_imp) for v in VIEWS}
    # sanidade: com a vista central, o professor melhorado tem que reproduzir o número já reportado
    assert abs(f1(yt, imp_t["center"]) - tr["teacher_improved_b5"]["f1_macro"]) < 1e-6

    md = json.loads((OUT / "megadetector" / "test_public.json").read_text())["detections"]
    edge = np.zeros(len(tids), dtype=bool)
    for k, iid in enumerate(tids):
        b = [d[2:] for d in md.get(iid, []) if d[0] == "animal" and d[1] >= MD_CONF]
        if b:
            x1, _, x2, _ = max(b, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            inside = max(0.0, min(x2, 1 - CROP_MARGIN) - max(x1, CROP_MARGIN))
            edge[k] = x2 > x1 and 1 - inside / (x2 - x1) >= 0.5

    rep = {}
    for name, probs in (("zero_shot", zs_t), ("melhorado", imp_t)):
        rep[name] = {}
        for h in hows:
            p = combine(probs, h).argmax(1)
            m = classification_metrics(yt, p)
            rep[name][h] = {"f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
                            "accuracy_animal_na_borda": float((p[edge] == yt[edge]).mean()),
                            "accuracy_animal_no_centro": float((p[~edge] == yt[~edge]).mean())}
    student = json.loads((OUT / "size_results.json").read_text())["center_crop"]
    res = {"beta": beta, "selection_105": sel, "chosen_on_105": chosen, "n_edge": int(edge.sum()),
           "test_public": rep,
           "aluno_referencia": {g: v["accuracy"]["aluno"] for g, v in student.items()}}
    (OUT / "teacher_multicrop.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"seleção nos 105 rótulos (F1): {sel}  -> escolhido: {chosen}")
    for name in rep:
        for h, r in rep[name].items():
            print(f"  {name:10s} {h:9s} F1 test {r['f1_macro']:.3f} | acc {r['accuracy']:.3f} | "
                  f"borda (n={edge.sum()}) {r['accuracy_animal_na_borda']:.2f} | centro {r['accuracy_animal_no_centro']:.2f}")
    print(f"  aluno (referência): {res['aluno_referencia']}")


if __name__ == "__main__":
    main()
