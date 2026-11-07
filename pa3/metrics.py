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


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
