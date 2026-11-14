"""Trilha C — o aluno: ResNet-18 (o mesmo backbone do piso) treinada nos
pseudo-rótulos do professor sobre o train inteiro, junto com os rótulos
verdadeiros do orçamento.

Variantes (uma ablação dos tratamentos de ruído, um de cada vez):
  hard_zs        professor zero-shot, rótulo duro (argmax) -- sem tratamento
  hard_improved  professor melhorado com os rótulos do orçamento, rótulo duro
  soft_improved  professor melhorado, rótulo suave (distribuição com a
                 temperatura calibrada nos rótulos do orçamento)
  soft_improved_tauXX  idem, descartando as imagens em que a confiança do
                 professor é < XX% (filtro por confiança; τ escolhido no val)
Em todas, as imagens rotuladas do orçamento entram com o rótulo verdadeiro
(e peso maior), no lugar do pseudo-rótulo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pa3.config import CKPT_DIR, NUM_CLASSES, OUTPUTS_DIR
from pa3.data import ImageArrayDataset, budget_labels, eval_dataset, get_cache, train_transform
from pa3.engine import evaluate, fit, predict_logits, set_seed
from pa3.models import build_resnet18, count_params, param_groups

RECIPE = dict(lr=2e-4, head_lr_mult=10.0, epochs=12, batch_size=32, weight_decay=1e-4, true_label_weight=5.0)

VARIANTS = {
    "hard_zs": dict(teacher="zs", soft=False, tau=0.0),
    "hard_improved": dict(teacher="improved", soft=False, tau=0.0),
    "soft_improved": dict(teacher="improved", soft=True, tau=0.0),
    "soft_improved_tau30": dict(teacher="improved", soft=True, tau=0.3),
    "soft_improved_tau50": dict(teacher="improved", soft=True, tau=0.5),
    "soft_improved_tau70": dict(teacher="improved", soft=True, tau=0.7),
}

# Pseudo-rótulos versionados no git (gerados por scripts/part1_teacher.py):
# assim quem treina o aluno (ex.: no Colab) não precisa rodar o professor.
PSEUDO_DIR = OUTPUTS_DIR / "part1" / "pseudo_labels"


def pseudo_label_path(teacher_variant: str, budget: int):
    return PSEUDO_DIR / f"pseudo_{teacher_variant}_b{budget}.npz"


def load_pseudo_labels(teacher_variant: str, budget: int) -> tuple[list[str], np.ndarray]:
    """(ids do train, probabilidades calibradas do professor, N x C)."""
    z = np.load(pseudo_label_path(teacher_variant, budget))
    return list(z["ids"]), z["probs"]


def build_targets(variant: str, budget: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    cfg = VARIANTS[variant]
    ids, probs = load_pseudo_labels(cfg["teacher"], budget)
    targets = probs.astype(np.float32) if cfg["soft"] else np.eye(NUM_CLASSES, dtype=np.float32)[probs.argmax(1)]
    weights = np.ones(len(ids), dtype=np.float32)
    idx = {iid: k for k, iid in enumerate(ids)}
    keep = probs.max(1) >= cfg["tau"]
    for iid, y in budget_labels(budget).items():
        k = idx[iid]
        targets[k] = np.eye(NUM_CLASSES, dtype=np.float32)[y]
        weights[k] = RECIPE["true_label_weight"]
        keep[k] = True  # rótulo verdadeiro nunca é filtrado
    keep = np.flatnonzero(keep)
    return [ids[k] for k in keep], targets[keep], weights[keep]


RUNS_DIR = OUTPUTS_DIR / "part1" / "students"


def train_student(variant: str, budget: int = 5, seed: int = 0, verbose: bool = True,
                  runs_dir: Path = RUNS_DIR, ckpt_dir: Path = CKPT_DIR / "student") -> dict:
    """Treina um aluno (ou, se a rodada já existe em `runs_dir`, só a lê).

    `runs_dir`/`ckpt_dir` diferentes dos padrões servem para retreinar sem
    tocar nos resultados versionados (scripts/train_final.py)."""
    tag = f"{variant}_b{budget}_s{seed}"
    if (runs_dir / f"{tag}.json").exists() and (runs_dir / f"{tag}_test_logits.npy").exists():
        if verbose:
            print(f"[aluno {tag}] já treinado, lendo {tag}.json", flush=True)
        return json.loads((runs_dir / f"{tag}.json").read_text())
    r = RECIPE
    set_seed(seed)
    ids, targets, weights = build_targets(variant, budget)
    train_ds = ImageArrayDataset(get_cache("train").subset(ids), targets, train_transform())
    model = build_resnet18(pretrained=True)
    out = fit(
        model,
        train_ds,
        eval_dataset("val"),
        param_groups=param_groups(model, "resnet18", r["lr"], r["head_lr_mult"]),
        epochs=r["epochs"],
        batch_size=r["batch_size"],
        weight_decay=r["weight_decay"],
        sample_weights=weights,
        log_prefix=f"[aluno {variant} b{budget} s{seed}] ",
        verbose=verbose,
        checkpoint_path=ckpt_dir / f"{variant}_b{budget}_s{seed}_last.pt",
    )
    model.load_state_dict(out["best_state"])
    test_ds = eval_dataset("test_public")
    logits = predict_logits(model, test_ds)
    test = evaluate(model, test_ds)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out["best_state"], ckpt_dir / f"{tag}.pt")
    result = {
        "variant": variant,
        "budget": budget,
        "seed": seed,
        "params": count_params(model),
        "n_train": len(ids),
        "best_epoch": out["best_epoch"],
        "best_val_f1": out["best_val_f1"],
        "test_f1_macro": test["f1_macro"],
        "test_accuracy": test["accuracy"],
        "test_f1_per_class": test["f1_per_class"],
        "history": out["history"],
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
    np.save(runs_dir / f"{tag}_test_logits.npy", logits)
    return result
