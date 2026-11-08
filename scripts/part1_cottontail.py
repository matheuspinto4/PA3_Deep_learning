"""Parte 1 (análise) — por que o aluno chama o coelho de gato?

O aluno escolhido perde ~0,67 de F1 para o professor em eastern_cottontail.
Duas hipóteses, testadas com vizinhos mais próximos no train:

1. Pseudo-rótulos ruins: se fosse isso, as imagens do train mais parecidas
   com os coelhos do teste no espaço do BioCLIP 2 (que olha o animal)
   estariam rotuladas como outra coisa.
2. Atalho de câmera: os coelhos do teste vêm de poucas câmeras; as fotos do
   train dessas mesmas câmeras (achadas por correlação de PIXELS, que mede o
   fundo) têm outros animais, e a CNN aprende "esta câmera = gato".

As câmeras do teste são agrupadas pela própria correlação em pixels. Os
vizinhos usam os pseudo-rótulos do professor melhorado (orçamento 5) que
passaram no filtro τ=0,7, ou seja, exatamente o que o aluno viu no treino.
Diagnóstico no test_public, nada é selecionado aqui.

Pressupõe os embeddings do professor (scripts/part1_teacher.py; se não
estiverem em cache/teacher/, são recalculados, o que exige o BioCLIP 2).

Saídas em outputs/part1/: 12_coelho_atalho_de_camera.png, cottontail_results.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pa3.config import CLASSES, OUTPUTS_DIR
from pa3.data import budget_labels, get_cache, is_night, split_labels
from pa3.plotting import COLORS, INK_MUTED, plt
from pa3.student import load_pseudo_labels
from pa3.teacher import cached_embeddings

OUT = OUTPUTS_DIR / "part1"
TARGET = "eastern_cottontail"
TAU = 0.7          # filtro do aluno escolhido
K = 10             # vizinhos no BioCLIP por imagem
CAMERA_SIM = 0.95  # correlação em pixels para agrupar coelhos do teste na mesma câmera
TRAIN_SIM = 0.93   # e para achar fotos do train da mesma câmera (conferido a olho: acima disso é a mesma cena)
CAMERA_NAMES = "ABCDEFGH"


def pixel_signature(cache, ids: list[str], chunk: int = 500) -> np.ndarray:
    """Foto em cinza reduzida a 28x40 (blocos de 8x8 px), centrada e
    normalizada: o produto interno vira a correlação entre as duas fotos.
    Captura o fundo (a câmera), não o animal pequeno."""
    out = []
    for b in range(0, len(ids), chunk):
        a = cache.subset(ids[b : b + chunk]).astype(np.float32).mean(-1)
        n, h, w = a.shape
        a = a.reshape(n, h // 8, 8, w // 8, 8).mean((2, 4)).reshape(n, -1)
        a -= a.mean(1, keepdims=True)
        out.append(a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6))
    return np.concatenate(out)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def group(sim: np.ndarray, thr: float) -> np.ndarray:
    """Agrupamento guloso: cada foto ainda sem grupo puxa as parecidas com ela."""
    g = -np.ones(len(sim), dtype=int)
    for i in range(len(sim)):
        if g[i] < 0:
            g[(sim[i] > thr) & (g < 0)] = g.max() + 1
    # renumera do maior grupo para o menor
    order = np.argsort(-np.bincount(g), kind="stable")
    return np.argsort(order)[g]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ct = CLASSES.index(TARGET)
    labels = split_labels("test_public")
    test_ids = sorted(labels)  # ordem de todos os arrays salvos
    y = np.array([labels[i] for i in test_ids])
    qmask = y == ct
    q = [i for i, m in zip(test_ids, qmask) if m]
    te_cache, tr_cache = get_cache("test_public"), get_cache("train")

    chosen = json.loads((OUT / "student_results.json").read_text())["chosen"]
    preds = {
        "piso b5": np.load(OUTPUTS_DIR / "floor_runs" / "resnet18_b5_s0_test_logits.npy").argmax(1),
        "piso b20": np.load(OUTPUTS_DIR / "floor_runs" / "resnet18_b20_s0_test_logits.npy").argmax(1),
        "professor melhorado": np.load(OUT / "teacher_test_probs_improved_b5.npy").argmax(1),
        "aluno b5": np.load(OUT / "students" / f"{chosen}_b5_s0_test_logits.npy").argmax(1),
        "aluno b20": np.load(OUT / "students" / f"{chosen}_b20_s0_test_logits.npy").argmax(1),
    }
    sq = preds["aluno b5"][qmask]  # previsão do aluno para cada coelho, na ordem de q
    res = {"student": chosen, "n_test": len(q), "frac_night": float(is_night(te_cache.subset(q)).mean()),
           "student_pred": {CLASSES[k]: int(v) for k, v in Counter(sq).most_common()},
           "recall": {k: float((p[qmask] == ct).mean()) for k, p in preds.items()}}

    # pseudo-rótulos que o aluno viu
    pids, P = load_pseudo_labels("improved", 5)
    pidx = {i: k for k, i in enumerate(pids)}
    parg, pconf = P.argmax(1), P.max(1)
    kept = pconf >= TAU

    def seen(train_rows) -> Counter:
        return Counter(CLASSES[parg[pidx[tr_ids[j]]]] for j in train_rows if kept[pidx[tr_ids[j]]])

    # --- hipótese 1: vizinhos no BioCLIP 2 ------------------------------------
    tr_ids, tr_emb = cached_embeddings("bioclip2", "center", "train")
    _, q_emb = cached_embeddings("bioclip2", "center", "test_public", ids=q)
    S = unit(q_emb) @ unit(tr_emb).T
    nn_bio = np.argsort(-S, axis=1)[:, :K]
    bio_seen = seen(nn_bio.ravel())
    res["bioclip_neighbors"] = {"k": K, "seen_by_student": dict(bio_seen.most_common()),
                                "frac_target": bio_seen[TARGET] / max(1, sum(bio_seen.values()))}

    # --- hipótese 2: câmeras --------------------------------------------------
    Q = pixel_signature(te_cache, q)
    R = pixel_signature(tr_cache, tr_ids)
    cam = group(Q @ Q.T, CAMERA_SIM)
    Sp = Q @ R.T
    budget = budget_labels(5)
    res["cameras"] = {}
    for c in range(cam.max() + 1):
        m = cam == c
        rows = np.where(Sp[m].max(0) > TRAIN_SIM)[0]
        sc = seen(rows)
        res["cameras"][CAMERA_NAMES[c]] = {
            "n_test_cottontails": int(m.sum()),
            "student_pred": {CLASSES[k]: int(v) for k, v in Counter(sq[m]).most_common()},
            "train_images_same_camera": int(len(rows)),
            "seen_by_student": dict(sc.most_common()),
            # o que o professor disse das fotos dessa câmera que o filtro τ jogou fora
            "filtered_out": dict(Counter(CLASSES[parg[pidx[tr_ids[j]]]] for j in rows
                                         if not kept[pidx[tr_ids[j]]]).most_common()),
            "budget_labels": [CLASSES[budget[tr_ids[j]]] for j in rows if tr_ids[j] in budget],
        }
    nn_pix = Sp.argmax(1)

    # --- figura ----------------------------------------------------------------
    # um coelho de cada câmera (até 4): fotos da mesma rajada são quase idênticas
    show = [int(np.where(cam == c)[0][0]) for c in range(min(4, cam.max() + 1))]
    ncol = len(show)
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, ncol, height_ratios=[1, 1, 1, 1.25], hspace=0.32, wspace=0.05)

    def pl(j):
        k = pidx[tr_ids[j]]
        return f"{CLASSES[parg[k]]} {pconf[k]:.2f}" + ("" if kept[k] else " (filtrado)")

    rows = [
        ("teste: coelho\n(previsão do aluno)",
         lambda k: (te_cache.subset([q[k]])[0], f"câmera {CAMERA_NAMES[cam[k]]} → {CLASSES[sq[k]]}")),
        ("vizinho no BioCLIP 2\n(pseudo-rótulo)",
         lambda k: (tr_cache.subset([tr_ids[nn_bio[k, 0]]])[0], pl(nn_bio[k, 0]))),
        ("vizinho em pixels\n(pseudo-rótulo, correlação)",
         lambda k: (tr_cache.subset([tr_ids[nn_pix[k]]])[0], f"{pl(nn_pix[k])} · r={Sp[k, nn_pix[k]]:.2f}")),
    ]
    for r, (name, get) in enumerate(rows):
        for col, k in enumerate(show):
            img, title = get(k)
            ax = fig.add_subplot(gs[r, col])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            # verde = coelho (certo), laranja = outra espécie
            ax.set_title(title, fontsize=9, color=COLORS["teacher"] if TARGET in title else COLORS["student"])
            if col == 0:
                ax.set_ylabel(name, fontsize=10)

    # (a) o que o aluno viu: vizinhos no BioCLIP vs. fotos da câmera A no train
    bottom = gs[3, :].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=0.4)
    ax1 = fig.add_subplot(bottom[0])
    camA = res["cameras"]["A"]
    cam_seen = Counter(camA["seen_by_student"])
    top = [c for c, _ in (bio_seen + cam_seen).most_common(6)]
    yy = np.arange(len(top))
    for off, (cnt, lbl, col) in zip((-0.2, 0.2), (
            (bio_seen, f"{K} vizinhos no BioCLIP 2 de cada coelho", COLORS["teacher"]),
            (cam_seen, f"fotos do train da câmera A (r > {TRAIN_SIM})", COLORS["student"]))):
        tot = max(1, sum(cnt.values()))
        vals = [cnt[c] / tot for c in top]
        bars = ax1.barh(yy + off, vals, 0.38, color=col, label=f"{lbl}, n={sum(cnt.values())}")
        ax1.bar_label(bars, labels=[f"{v:.0%}" if v > 0 else "" for v in vals], fontsize=8, padding=2)
    ax1.set_yticks(yy, top)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.12)
    ax1.set_xlabel(f"fração dos pseudo-rótulos que o aluno viu (τ ≥ {TAU})")
    ax1.set_title("O que o aluno aprendeu sobre essas fotos")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(axis="y", visible=False)

    # (b) quem acerta os coelhos
    ax2 = fig.add_subplot(bottom[1])
    cols = {"piso": COLORS["floor"], "professor": COLORS["teacher"], "aluno": COLORS["student"]}
    names = list(res["recall"])
    bars = ax2.bar(range(len(names)), [res["recall"][k] for k in names], 0.6,
                   color=[cols[k.split()[0]] for k in names])
    ax2.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax2.set_xticks(range(len(names)), [k.replace(" ", "\n", 1) for k in names], fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel(f"recall de {TARGET} (n={len(q)})")
    ax2.set_title("Quem acerta os coelhos")
    ax2.grid(axis="x", visible=False)

    summary = "; ".join(f"câmera {c}: {v['n_test_cottontails']} coelhos → "
                        + ", ".join(f"{k} {n}" for k, n in v["student_pred"].items())
                        for c, v in res["cameras"].items())
    fig.text(0.5, 0.905, summary, ha="center", fontsize=9.5, color=INK_MUTED)
    fig.suptitle("Atalho de câmera: para os coelhos, o aluno responde pela câmera, não pelo animal",
                 fontweight="bold", y=0.93)
    fig.savefig(OUT / "12_coelho_atalho_de_camera.png")
    plt.close(fig)
    (OUT / "cottontail_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(q)} coelhos no teste ({res['frac_night']:.0%} noturnos); aluno prevê: {res['student_pred']}")
    print("recall: " + " | ".join(f"{k} {v:.2f}" for k, v in res["recall"].items()))
    print(f"BioCLIP: pseudo-rótulos vistos pelo aluno entre os {K} vizinhos: {dict(bio_seen.most_common(4))} "
          f"({res['bioclip_neighbors']['frac_target']:.0%} coelho)")
    for c, v in res["cameras"].items():
        print(f"câmera {c}: {v['n_test_cottontails']} coelhos -> aluno {v['student_pred']} | "
              f"{v['train_images_same_camera']} fotos no train, aluno viu {v['seen_by_student']}, "
              f"filtradas {v['filtered_out']}, "
              f"orçamento: {v['budget_labels']}")


if __name__ == "__main__":
    main()
