"""O piso: supervisionado só com os rótulos do orçamento (Parte 0, e Parte 2).

Uma única receita por arquitetura, usada sem mudança em todos os orçamentos
e seeds -- assim a curva de orçamento compara o mesmo procedimento.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from pa3.config import CKPT_DIR, OUTPUTS_DIR
from pa3.data import budget_dataset, eval_dataset, eval_transform, train_transform
from pa3.engine import evaluate, fit, predict_logits, set_seed
from pa3.models import build_model, count_params, param_groups

RECIPES = {
    # CNN do zero: lr maior, mais épocas (não há nada pré-aprendido para preservar).
    "small_cnn": dict(lr=1e-3, head_lr_mult=1.0, epochs={1: 200, 5: 120, 20: 60}, batch_size=16, weight_decay=5e-4),
    # Fine-tuning: backbone com lr 10x menor que a cabeça nova.
    "resnet18": dict(lr=1e-4, head_lr_mult=10.0, epochs={1: 120, 5: 40, 20: 30}, batch_size=16, weight_decay=1e-4),
}


RUNS_DIR = OUTPUTS_DIR / "floor_runs"


def train_floor(arch: str, budget: int = 5, seed: int = 0, verbose: bool = True) -> dict:
    """Treina (ou, se a rodada já existe em disco, só recarrega) um piso."""
    tag = f"{arch}_b{budget}_s{seed}"
    ckpt, res_path, logits_path = CKPT_DIR / "floor" / f"{tag}.pt", RUNS_DIR / f"{tag}.json", RUNS_DIR / f"{tag}_test_logits.npy"
    if ckpt.exists() and res_path.exists() and logits_path.exists():
        if verbose:
            print(f"[{tag}] já treinado, recarregando de {res_path.name}", flush=True)
        model = build_model(arch, pretrained=False)
        model.load_state_dict(torch.load(ckpt))
        return {"model": model, "result": json.loads(res_path.read_text()), "test_logits": np.load(logits_path),
                "test_labels": np.asarray(eval_dataset("test_public").labels)}

    r = RECIPES[arch]
    set_seed(seed)
    model = build_model(arch, pretrained=True)
    train_ds = budget_dataset(budget, train_transform())
    out = fit(
        model,
        train_ds,
        eval_dataset("val"),
        param_groups=param_groups(model, arch, r["lr"], r["head_lr_mult"]),
        epochs=r["epochs"][budget],
        batch_size=r["batch_size"],
        weight_decay=r["weight_decay"],
        train_eval_ds=budget_dataset(budget, eval_transform()),
        log_prefix=f"[{arch} b{budget} s{seed}] ",
        verbose=verbose,
        checkpoint_path=CKPT_DIR / "floor" / f"{tag}_last.pt",
    )
    model.load_state_dict(out["best_state"])
    test_ds = eval_dataset("test_public")
    logits = predict_logits(model, test_ds)
    test = evaluate(model, test_ds)

    (CKPT_DIR / "floor").mkdir(parents=True, exist_ok=True)
    torch.save(out["best_state"], ckpt)
    result = {
        "arch": arch,
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
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(result, indent=2))
    np.save(logits_path, logits)
    return {"model": model, "result": result, "test_logits": logits, "test_labels": np.asarray(test_ds.labels)}


def save_result(result: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
