"""Laço de treino supervisionado com seleção de época no val.

O mesmo `fit` serve ao piso (Parte 0/2) e ao aluno da Trilha C: os rótulos
podem ser classes duras (int) ou distribuições (rótulos suaves do professor),
e cada amostra pode ter um peso (ex.: confiança do pseudo-rótulo).
"""

from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pa3.metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Entropia cruzada por amostra, aceitando alvo inteiro ou distribuição."""
    if target.dim() == 1:
        return F.cross_entropy(logits, target, reduction="none", label_smoothing=label_smoothing)
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1)


@torch.no_grad()
def predict_logits(model: nn.Module, ds: Dataset, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = [model(x) for x, _ in DataLoader(ds, batch_size=batch_size, shuffle=False)]
    return torch.cat(out).numpy()


def evaluate(model: nn.Module, ds: Dataset) -> dict:
    logits = predict_logits(model, ds)
    y = np.asarray(ds.labels)
    m = classification_metrics(y, logits.argmax(1))
    m["loss"] = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y)).item())
    return m


def _cosine_with_warmup(total_steps: int, warmup_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    return f


def fit(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    param_groups: list[dict],
    epochs: int,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    warmup_epochs: int = 1,
    sample_weights: np.ndarray | None = None,
    train_eval_ds: Dataset | None = None,
    log_prefix: str = "",
    verbose: bool = True,
    checkpoint_path=None,
) -> dict:
    """Treina `epochs` épocas e devolve o histórico e o estado da melhor época.

    Melhor época = maior F1 macro no val; empate decidido pela menor perda no
    val. Nada do test_public entra aqui.

    `train_eval_ds` (opcional): o próprio conjunto de treino sem aumentação,
    avaliado a cada época -- é ele que mostra quando a rede passa a decorar.

    `checkpoint_path` (opcional): ao fim de cada época salva modelo,
    otimizador, scheduler, RNG, histórico e melhor estado; se o arquivo já
    existe, o treino continua da época seguinte em vez de recomeçar. O arquivo
    é apagado quando o treino termina.
    """
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _cosine_with_warmup(epochs * steps_per_epoch, warmup_epochs * steps_per_epoch)
    )
    weights = torch.ones(len(train_ds)) if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.float32)
    loader = DataLoader(_Indexed(train_ds), batch_size=batch_size, shuffle=True, drop_last=False)

    history, best = [], {"f1": -1.0, "loss": float("inf"), "epoch": -1, "state": None}
    start = 1
    if checkpoint_path is not None and checkpoint_path.exists():
        ck = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["np_rng"])
        random.setstate(ck["py_rng"])
        history, best, start = ck["history"], ck["best"], ck["epoch"] + 1
        if verbose:
            print(f"{log_prefix}retomando do checkpoint da época {ck['epoch']}", flush=True)

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss, n = 0.0, 0
        for x, y, idx in loader:
            w = weights[idx]
            loss = (soft_cross_entropy(model(x), y, label_smoothing) * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(x)
            n += len(x)

        val = evaluate(model, val_ds)
        row = {"epoch": epoch, "train_loss_aug": tot_loss / n, "val_loss": val["loss"], "val_f1": val["f1_macro"], "val_acc": val["accuracy"]}
        if train_eval_ds is not None:
            tr = evaluate(model, train_eval_ds)
            row.update(train_loss=tr["loss"], train_f1=tr["f1_macro"], train_acc=tr["accuracy"])
        row["seconds"] = time.time() - t0
        history.append(row)

        if val["f1_macro"] > best["f1"] or (val["f1_macro"] == best["f1"] and val["loss"] < best["loss"]):
            best = {"f1": val["f1_macro"], "loss": val["loss"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if verbose:
            extra = f" | train F1 {row['train_f1']:.3f}" if "train_f1" in row else ""
            print(
                f"{log_prefix}ep {epoch:3d} | loss {row['train_loss_aug']:.3f}{extra} | "
                f"val loss {val['loss']:.3f} F1 {val['f1_macro']:.3f} acc {val['accuracy']:.3f} | {row['seconds']:.0f}s",
                flush=True,
            )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(), "py_rng": random.getstate(),
                 "history": history, "best": best, "epoch": epoch},
                tmp,
            )
            tmp.replace(checkpoint_path)  # escrita atômica: uma queda no meio não corrompe o checkpoint

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()
    return {"history": history, "best_epoch": best["epoch"], "best_val_f1": best["f1"], "best_state": best["state"]}


class _Indexed(Dataset):
    """Devolve também o índice da amostra (para buscar o peso dela)."""

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i
