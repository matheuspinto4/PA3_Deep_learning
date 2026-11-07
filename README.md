# PA3 — Poucos rótulos: fauna em câmeras trap

Classificação de espécie (21 classes, ENA24) com **5 rótulos por classe** e
6.505 imagens sem rótulo. Métrica principal: **F1 macro** no `test_public`.
Ver `PA3.pdf` para o enunciado completo.

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

Tudo foi desenvolvido e rodado **em CPU** (sem GPU: i5-1240P, 16 GB), com
PyTorch 2.14 CPU. Os dados (`pa3_dados.zip`) devem ser extraídos em
`pa3_dados/pa3_dados/` (fora do git).

## Estrutura

```
pa3/
  config.py      # caminhos, classes, resolução de entrada (224x320), teto
  data.py        # CSVs de rótulo, cache de imagens, aumentação, detector de foto noturna
  models.py      # CNN pequena do zero e ResNet-18 ImageNet (contagem de parâmetros)
  engine.py      # laço de treino (seleção de época no val, rótulos suaves, checkpoint por época)
  metrics.py     # F1 macro, acurácia, F1 por classe, matriz de confusão
  floor.py       # receita do piso (Parte 0, reutilizada na Parte 2)
  teacher.py     # Trilha C: professor (BioCLIP/CLIP), prompts, professor melhorado
  student.py     # Trilha C: aluno ResNet-18 nos pseudo-rótulos (3 variantes)
  plotting.py    # estilo e matriz de confusão
scripts/
  build_cache.py      # pré-redimensiona as imagens uma vez (cache/)
  part0_floor.py      # Parte 0: o piso
  part1_teacher.py    # Parte 1: escolha e análise do professor + pseudo-rótulos
  part1_student.py    # Parte 1: alunos, comparação com professor e piso
outputs/partN/        # figuras e números de cada parte
outputs/floor_runs/   # cada rodada do piso (json + logits de teste)
logs/                 # saída dos treinos
AI_LOG.md             # uso de IA no projeto
```

## Comandos

```bash
python scripts/build_cache.py      # uma vez (~2 min)
python scripts/part0_floor.py      # Parte 0 (~35 min em CPU)
python scripts/part1_teacher.py    # Parte 1: professor (~20 min; embeddings ficam em cache/)
python scripts/part1_student.py    # Parte 1: 3 alunos (~1 h cada)
```

Todos os treinos salvam checkpoint a cada época: rodar o mesmo comando de
novo continua de onde parou, e rodadas já concluídas são recarregadas em vez
de retreinadas.

## Decisões registradas

- **Trilha da Parte 1:** C (professor e aluno). Sem GPU, é a única trilha em
  que a parte cara (o *foundation model*) é só inferência, feita uma vez.
- **Backbone:** ResNet-18 ImageNet (11.187.285 parâmetros com a cabeça de 21
  classes, abaixo do limite de 12M). É o mesmo backbone do teto (0,686).
- **Entrada:** quadro inteiro redimensionado para 224×320 (altura×largura),
  sem center crop, para não cortar animais pequenos na borda.
- **Seleção de modelo:** só no `val` (F1 macro; empate pela perda de val). O
  `test_public` só é avaliado depois, para reportar.
- **O professor nunca vê o `val`:** a escolha do professor (modelo, vista,
  prompts) e o ajuste do professor melhorado usam só os rótulos do orçamento,
  que são do `train`.

## Parte 0 — O piso

Supervisionado só com `rotulos_5.csv` (105 imagens), seed 0. Época escolhida
pelo F1 macro no val.

| versão | parâmetros | melhor época (val) | F1 val | **F1 macro test** | acurácia test |
|---|---|---|---|---|---|
| (a) CNN pequena do zero | 1.187.893 | 120 | 0,360 | **0,307** | 0,353 |
| (b) ResNet-18 ImageNet + fine-tuning | 11.187.285 | 13 | 0,424 | **0,460** | 0,485 |
| teto (monitor: ResNet-18, todos os rótulos do train) | | | | 0,686 | |

**Em que época começa a decorar?** (`outputs/part0/03_curva_resnet18.png`)
A ResNet-18 acerta ≥99% das 105 imagens de treino (sem aumentação) já na
**época 9**. A partir daí a perda de treino vai a ~0, enquanto o F1 de val fica
num platô de 0,37–0,42 e a perda de val mal se mexe (2,04 na época 6, mínimo
de 2,00 na época 24). A rede decora as 105 imagens em ~9 épocas, e todo
o treino seguinte só aumenta a confiança no treino, sem ganho real no val.

Figuras: `01_piso_f1_acuracia.png`, `02_matriz_confusao_resnet18.png`,
`03_curva_resnet18.png`, `03b_curva_small_cnn.png`, `04_f1_por_classe.png`.
