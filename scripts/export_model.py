"""Empacota um checkpoint de treino no formato do modelo entregue.

    python scripts/export_model.py CHECKPOINT [--saida weights/modelo_final.pt]

Salva state_dict + nomes das classes (ordem de classes.csv) + resolução de
entrada, para o predict.py rodar sem o pacote de dados. Confere o limite de
12M parâmetros do enunciado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from pa3.config import CLASSES, IMG_H, IMG_W
from pa3.models import MAX_PARAMS, build_resnet18, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--saida", type=Path, default=ROOT / "weights" / "modelo_final.pt")
    ap.add_argument("--origem", default="", help="descrição de onde veio o modelo (vai para o arquivo)")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    model = build_resnet18(pretrained=False)
    model.load_state_dict(state)
    n = count_params(model)
    assert n <= MAX_PARAMS, f"{n:,} parâmetros > limite de {MAX_PARAMS:,}"
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "input_hw": [IMG_H, IMG_W],
                "arch": "resnet18", "params": n, "origem": args.origem or args.checkpoint.name}, args.saida)
    print(f"{args.saida} ({args.saida.stat().st_size / 1e6:.1f} MB, {n:,} parâmetros)")


if __name__ == "__main__":
    main()
