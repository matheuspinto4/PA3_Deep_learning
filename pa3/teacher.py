"""Trilha C — o professor (foundation model aberto, rodando local).

Regras do enunciado que este módulo respeita:
  - o professor só vê `train` e `test_public` (este só para medi-lo);
    NUNCA o `val` -- não existe função aqui que leia o val;
  - ele não faz parte do modelo entregue (só gera pseudo-rótulos).

Todo o trabalho caro (passar 7.3k imagens por um ViT em CPU) é feito UMA vez:
os embeddings de imagem ficam em `cache/teacher/`. Prompt, temperatura e o
professor melhorado com os rótulos do orçamento são contas baratas em cima
desses embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pa3.config import CACHE_DIR, CLASSES, IMAGES_DIR

TEACHERS = {
    "clip_b32": dict(model_name="ViT-B-32", pretrained="openai"),
    "bioclip": dict(model_name="hf-hub:imageomics/bioclip"),
    "bioclip2": dict(model_name="hf-hub:imageomics/bioclip-2"),
}

# Nome comum + científico. O nome científico importa para o BioCLIP, que foi
# treinado com a taxonomia (TreeOfLife-10M) no texto.
SPECIES = {
    "american_black_bear": ("American black bear", "Ursus americanus"),
    "american_crow": ("American crow", "Corvus brachyrhynchos"),
    "bird": ("bird", "Aves"),
    "bobcat": ("bobcat", "Lynx rufus"),
    "chicken": ("chicken", "Gallus gallus domesticus"),
    "coyote": ("coyote", "Canis latrans"),
    "dog": ("dog", "Canis familiaris"),
    "domestic_cat": ("domestic cat", "Felis catus"),
    "eastern_chipmunk": ("eastern chipmunk", "Tamias striatus"),
    "eastern_cottontail": ("eastern cottontail rabbit", "Sylvilagus floridanus"),
    "eastern_fox_squirrel": ("eastern fox squirrel", "Sciurus niger"),
    "eastern_gray_squirrel": ("eastern gray squirrel", "Sciurus carolinensis"),
    "grey_fox": ("gray fox", "Urocyon cinereoargenteus"),
    "horse": ("horse", "Equus caballus"),
    "northern_raccoon": ("raccoon", "Procyon lotor"),
    "red_fox": ("red fox", "Vulpes vulpes"),
    "striped_skunk": ("striped skunk", "Mephitis mephitis"),
    "virginia_opossum": ("Virginia opossum", "Didelphis virginiana"),
    "white_tailed_deer": ("white-tailed deer", "Odocoileus virginianus"),
    "wild_turkey": ("wild turkey", "Meleagris gallopavo"),
    "woodchuck": ("woodchuck", "Marmota monax"),
}

# "basic" = o que o enunciado pede como linha de referência: o nome da classe
# do classes.csv, sem engenharia. "ensemble" = média de vários templates com
# nome comum e científico, incluindo a cena de câmera trap e a foto noturna.
PROMPT_SETS = {
    "basic": ["{label}"],
    "ensemble": [
        "a photo of a {common}.",
        "a camera trap photo of a {common}.",
        "a camera trap photo of a {common} at night.",
        "an infrared night photo of a {common}.",
        "a blurry photo of a {common}.",
        "a photo of a small {common} far away.",
        "a photo of {sci}.",
        "a camera trap photo of a {common}, {sci}.",
    ],
}


def class_prompts(prompt_set: str) -> list[list[str]]:
    out = []
    for c in CLASSES:
        common, sci = SPECIES[c]
        out.append([t.format(label=c.replace("_", " "), common=common, sci=sci) for t in PROMPT_SETS[prompt_set]])
    return out


def load_teacher(name: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(**TEACHERS[name])
    tokenizer = open_clip.get_tokenizer(TEACHERS[name]["model_name"])
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def text_classifier(model, tokenizer, prompt_set: str) -> np.ndarray:
    """(C, D): um vetor de texto normalizado por classe (média dos templates)."""
    rows = []
    for prompts in class_prompts(prompt_set):
        e = F.normalize(model.encode_text(tokenizer(prompts)), dim=-1).mean(0)
        rows.append(F.normalize(e, dim=0))
    return torch.stack(rows).numpy()


def _letterbox(img: Image.Image) -> Image.Image:
    """Completa a imagem até ficar quadrada (bordas cinza) em vez de cortar."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (124, 116, 104))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


@torch.no_grad()
def encode_images(model, preprocess, ids: list[str], split: str, view: str = "center", batch_size: int = 32, log_every: int = 20) -> np.ndarray:
    """Embeddings normalizados das imagens originais (512px, não o cache).

    view="center": pré-processamento padrão do CLIP (lado menor 224 + center
    crop) -- joga fora as laterais da foto.
    view="letterbox": a foto inteira, com faixas cinza para ficar quadrada.
    """
    feats = []
    for b in range(0, len(ids), batch_size):
        imgs = []
        for iid in ids[b : b + batch_size]:
            img = Image.open(IMAGES_DIR / split / f"{iid}.jpg").convert("RGB")
            imgs.append(preprocess(_letterbox(img) if view == "letterbox" else img))
        feats.append(F.normalize(model.encode_image(torch.stack(imgs)), dim=-1))
        if log_every and (b // batch_size) % log_every == 0:
            print(f"  {split}/{view}: {b + len(imgs)}/{len(ids)}", flush=True)
    return torch.cat(feats).numpy()


def embedding_path(teacher: str, view: str, split: str):
    d = CACHE_DIR / "teacher"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{teacher}_{view}_{split}.npz"


def cached_embeddings(teacher: str, view: str, split: str, ids: list[str] | None = None, model_bundle=None) -> tuple[list[str], np.ndarray]:
    """Embeddings de um split inteiro (ou de `ids`), calculados na 1ª chamada."""
    assert split in ("train", "test_public"), "o professor nunca olha o val"
    path = embedding_path(teacher, view, split)
    if path.exists():
        z = np.load(path)
        all_ids, emb = list(z["ids"]), z["emb"]
        if ids is None:
            return all_ids, emb
        idx = {i: k for k, i in enumerate(all_ids)}
        if all(i in idx for i in ids):
            return ids, emb[[idx[i] for i in ids]]
    if ids is None:
        ids = sorted(p.stem for p in (IMAGES_DIR / split).glob("*.jpg"))
        model, preprocess, _ = model_bundle or load_teacher(teacher)
        emb = encode_images(model, preprocess, ids, split, view)
        np.savez(path, ids=np.array(ids), emb=emb)
        return ids, emb
    # subconjunto pedido (ex.: as 105 imagens da escolha do professor) e sem
    # cache do split inteiro: usa/gera um cache só desse subconjunto
    sub_path = path.with_name(path.stem + f"_subset{len(ids)}.npz")
    if sub_path.exists():
        z = np.load(sub_path)
        if list(z["ids"]) == list(ids):
            return ids, z["emb"]
    model, preprocess, _ = model_bundle or load_teacher(teacher)
    emb = encode_images(model, preprocess, ids, split, view, log_every=0)
    np.savez(sub_path, ids=np.array(ids), emb=emb)
    return ids, emb


# ----------------------------------------------------------------------------
# Zero-shot, calibração e professor melhorado com os rótulos do orçamento
# ----------------------------------------------------------------------------

LOGIT_SCALE = 100.0  # escala do CLIP: logits = 100 * cosseno


def zero_shot_logits(img_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    return LOGIT_SCALE * img_emb @ text_emb.T


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def prototype_classifier(text_emb: np.ndarray, img_emb: np.ndarray, labels: np.ndarray, beta: float) -> np.ndarray:
    """Professor melhorado: o protótipo de cada classe vira
    normalize(texto + beta * média dos embeddings das imagens rotuladas).
    beta=0 é o zero-shot puro; beta grande é um classificador de centróide
    só com as imagens. Classes sem imagem rotulada ficam só com o texto."""
    protos = text_emb.copy()
    for c in range(len(text_emb)):
        m = labels == c
        if m.any():
            mean = img_emb[m].mean(0)
            protos[c] = text_emb[c] + beta * mean / np.linalg.norm(mean)
    return protos / np.linalg.norm(protos, axis=1, keepdims=True)


def loo_prototype_predictions(text_emb: np.ndarray, img_emb: np.ndarray, labels: np.ndarray, beta: float) -> np.ndarray:
    """Logits leave-one-out nas próprias imagens rotuladas: cada imagem é
    classificada por protótipos montados SEM ela. É assim que escolhemos beta
    e a temperatura sem tocar no val."""
    out = np.zeros((len(labels), len(text_emb)))
    for i in range(len(labels)):
        keep = np.arange(len(labels)) != i
        protos = prototype_classifier(text_emb, img_emb[keep], labels[keep], beta)
        out[i] = LOGIT_SCALE * img_emb[i] @ protos.T
    return out


def fit_temperature(logits: np.ndarray, labels: np.ndarray, grid=None) -> float:
    """Temperatura que minimiza a NLL (calibração) nos rótulos do orçamento."""
    grid = grid if grid is not None else np.geomspace(0.25, 8, 41)
    nll = [-np.log(softmax(logits, T)[np.arange(len(labels)), labels] + 1e-12).mean() for T in grid]
    return float(grid[int(np.argmin(nll))])
