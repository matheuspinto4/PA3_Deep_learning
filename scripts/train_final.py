"""Treina do zero o modelo entregue e exporta os pesos.

    python scripts/train_final.py [--saida weights/modelo_retreinado.pt]

É o aluno escolhido no val na Parte 1 (soft_improved_tau70: rótulos suaves do
professor melhorado, filtro de confiança τ=0,7, orçamento 5, seed 0), com a
mesma receita. Usa os pseudo-rótulos versionados em
outputs/part1/pseudo_labels/ (não precisa do professor) e o cache de imagens
(scripts/build_cache.py).

Grava em pastas próprias (outputs/final_retrain/, checkpoints/final_retrain/)
para sempre treinar de verdade e nunca sobrescrever os resultados
versionados. Por padrão também não sobrescreve weights/modelo_final.pt.
Em GPU o treino usa precisão mista, que não é determinística: o F1 obtido
varia na ordem do desvio entre seeds (~0,02) em torno do reportado (0,683).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_model import export

from pa3.config import CKPT_DIR, DEVICE, OUTPUTS_DIR
from pa3.student import train_student

VARIANT, BUDGET, SEED = "soft_improved_tau70", 5, 0
RUNS_DIR = OUTPUTS_DIR / "final_retrain"
RUN_CKPT_DIR = CKPT_DIR / "final_retrain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=ROOT / "weights" / "modelo_retreinado.pt")
    args = ap.parse_args()

    tag = f"{VARIANT}_b{BUDGET}_s{SEED}"
    for f in (RUNS_DIR / f"{tag}.json", RUNS_DIR / f"{tag}_test_logits.npy"):
        f.unlink(missing_ok=True)  # resultado de um retreino anterior: treina de novo
    print(f"treinando {tag} em {DEVICE}")
    r = train_student(VARIANT, BUDGET, SEED, runs_dir=RUNS_DIR, ckpt_dir=RUN_CKPT_DIR)
    print(f"época escolhida no val: {r['best_epoch']} (F1 val {r['best_val_f1']:.3f}) | "
          f"test_public: F1 {r['test_f1_macro']:.3f}, acurácia {r['test_accuracy']:.3f} (reportado: 0,683 / 0,736)")
    export(RUN_CKPT_DIR / f"{tag}.pt", args.saida, origem=f"{tag}, retreinado por scripts/train_final.py")


if __name__ == "__main__":
    main()
