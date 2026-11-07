"""Parte 1 (análise) — tamanho do animal com o MegaDetector V6.

O pacote não traz bounding box, mas o enunciado pede para dizer onde o
professor erra com "animais pequenos". O MegaDetector (detector de animal /
pessoa / veículo treinado em câmeras trap) entra aqui só como instrumento de
medida, como a Trilha C permite: roda no train e no test_public, nunca no val,
e não faz parte do modelo entregue nem gera rótulo de espécie.

Saída: outputs/part1/megadetector/<split>.json com, por imagem, as detecções
[classe, confiança, x1, y1, x2, y2] em coordenadas normalizadas (0-1).
A mesma caixa serve na Parte 3 ("o animal ocupa 40x30 px depois do resize").

Requer `ultralytics` (ver colab/run_pa3.ipynb; no Windows desta máquina o
OpenCV que ele usa não carrega, por isso este script roda no Colab).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pa3.config import CKPT_DIR, IMAGES_DIR, OUTPUTS_DIR

WEIGHTS_URL = "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1"
WEIGHTS = CKPT_DIR / "megadetector" / "MDV6-yolov9-c.pt"
OUT = OUTPUTS_DIR / "part1" / "megadetector"
MIN_CONF = 0.1  # guarda tudo acima de 0.1; o limiar de uso é decidido na análise
SPLITS = ("test_public", "train")


def main():
    from ultralytics import YOLO

    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        print("baixando pesos do MegaDetector V6 (yolov9-c)...", flush=True)
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    model = YOLO(str(WEIGHTS))
    names = model.names
    print("classes do detector:", names, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        path = OUT / f"{split}.json"
        if path.exists():
            print(f"{split}: já existe, pulando")
            continue
        paths = sorted((IMAGES_DIR / split).glob("*.jpg"))
        dets = {}
        for b in range(0, len(paths), 64):
            batch = [str(p) for p in paths[b : b + 64]]
            for p, r in zip(batch, model.predict(batch, conf=MIN_CONF, imgsz=640, verbose=False)):
                boxes = r.boxes
                xyxyn = boxes.xyxyn.cpu().numpy().round(4).tolist()
                dets[Path(p).stem] = [
                    [names[int(c)], round(float(s), 3), *xy] for c, s, xy in zip(boxes.cls.tolist(), boxes.conf.tolist(), xyxyn)
                ]
            print(f"  {split}: {min(b + 64, len(paths))}/{len(paths)}", flush=True)
        path.write_text(json.dumps({"model": "MDV6-yolov9-c", "min_conf": MIN_CONF, "format": "[classe, conf, x1, y1, x2, y2] normalizado", "detections": dets}))
        n_animal = sum(any(d[0] == "animal" and d[1] >= 0.2 for d in v) for v in dets.values())
        print(f"{split}: {len(dets)} imagens, {n_animal} com animal (conf>=0.2)", flush=True)


if __name__ == "__main__":
    main()
