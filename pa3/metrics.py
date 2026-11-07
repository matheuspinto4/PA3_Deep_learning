"""F1 macro (métrica principal), acurácia, F1 por classe e matriz de confusão."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from pa3.config import NUM_CLASSES

ALL_LABELS = list(range(NUM_CLASSES))


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # labels=ALL_LABELS: uma classe que o modelo nunca prevê conta como F1=0
    # no macro, em vez de sumir da média.
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "f1_per_class": f1_score(y_true, y_pred, labels=ALL_LABELS, average=None, zero_division=0).tolist(),
    }


def subset_metrics(y_true, y_pred) -> dict:
    """Métricas num SUBCONJUNTO (ex.: só fotos noturnas). Aqui o F1 macro é
    feito só sobre as classes presentes no subconjunto: com labels=todas, uma
    classe que nem aparece (não há gambá de dia) entraria como F1=0 e
    derrubaria a média sem dizer nada sobre o modelo."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = np.unique(y_true)
    return {
        "n": int(len(y_true)),
        "n_classes": int(len(present)),
        "accuracy": float((y_true == y_pred).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
    }


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
