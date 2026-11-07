"""Trilha C — o aluno: ResNet-18 (o mesmo backbone do piso) treinada nos
pseudo-rótulos do professor sobre o train inteiro, junto com os rótulos
verdadeiros do orçamento.

Variantes (uma ablação de dois tratamentos de ruído):
  hard_zs        professor zero-shot, rótulo duro (argmax) -- sem tratamento
  hard_improved  professor melhorado com os rótulos do orçamento, rótulo duro
  soft_improved  professor melhorado, rótulo suave (distribuição com a
                 temperatura calibrada nos rótulos do orçamento)
Nos três, as imagens rotuladas do orçamento entram com o rótulo verdadeiro
(e peso maior), no lugar do pseudo-rótulo.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from pa3.config import CACHE_DIR, CKPT_DIR, NUM_CLASSES, OUTPUTS_DIR
from pa3.data import ImageArrayDataset, budget_labels, eval_dataset, get_cache, train_transform
from pa3.engine import evaluate, fit, predict_logits, set_seed
from pa3.models import build_resnet18, count_params, param_groups

RECIPE = dict(lr=2e-4, head_lr_mult=10.0, epochs=8, batch_size=32, weight_decay=1e-4, true_label_weight=5.0)

VARIANTS = {
    "hard_zs": dict(teacher="zs", soft=False),
    "hard_improved": dict(teacher="improved", soft=False),
    "soft_improved": dict(teacher="improved", soft=True),
}


def pseudo_label_path(teacher_variant: str, budget: int):
    return CACHE_DIR / "teacher" / f"pseudo_{teacher_variant}_b{budget}.npz"


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
    for iid, y in budget_labels(budget).items():
        k = idx[iid]
        targets[k] = np.eye(NUM_CLASSES, dtype=np.float32)[y]
        weights[k] = RECIPE["true_label_weight"]
    return ids, targets, weights


def train_student(variant: str, budget: int = 5, seed: int = 0, verbose: bool = True) -> dict:
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
        checkpoint_path=CKPT_DIR / "student" / f"{variant}_b{budget}_s{seed}_last.pt",
    )
    model.load_state_dict(out["best_state"])
    test_ds = eval_dataset("test_public")
    logits = predict_logits(model, test_ds)
    test = evaluate(model, test_ds)

    tag = f"{variant}_b{budget}_s{seed}"
    (CKPT_DIR / "student").mkdir(parents=True, exist_ok=True)
    torch.save(out["best_state"], CKPT_DIR / "student" / f"{tag}.pt")
    result = {
        "variant": variant,
        "budget": budget,
        "seed": seed,
        "params": count_params(model),
        "best_epoch": out["best_epoch"],
        "best_val_f1": out["best_val_f1"],
        "test_f1_macro": test["f1_macro"],
        "test_accuracy": test["accuracy"],
        "test_f1_per_class": test["f1_per_class"],
        "history": out["history"],
    }
    d = OUTPUTS_DIR / "part1" / "students"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.json").write_text(json.dumps(result, indent=2))
    np.save(d / f"{tag}_test_logits.npy", logits)
    return result
