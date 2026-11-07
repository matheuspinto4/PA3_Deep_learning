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
python scripts/build_cache.py         # uma vez (~2 min)
python scripts/part0_floor.py         # Parte 0 (~35 min em CPU)
python scripts/part1_teacher.py       # Parte 1: professor (~2 h em CPU na 1a vez; depois, segundos)
python scripts/part1_megadetector.py  # Parte 1: tamanho dos animais (requer ultralytics)
python scripts/part1_student.py       # Parte 1: 6 alunos (GPU recomendada)
python scripts/part2_budget.py        # Parte 2: curva de orçamento (GPU recomendada)
```

Todos os treinos salvam checkpoint a cada época: rodar o mesmo comando de
novo continua de onde parou, e rodadas já concluídas são recarregadas em vez
de retreinadas.

**Onde cada coisa rodou.** Parte 0 e o professor rodaram em CPU, nesta
máquina. Os alunos, o MegaDetector e a Parte 2 rodaram no Colab (GPU T4) com
`colab/run_pa3.ipynb`, que clona este repositório, usa os pseudo-rótulos
versionados em `outputs/part1/pseudo_labels/` e grava os resultados no Drive.
Em CPU, cada aluno levava ~1h20 (6,5k imagens × 12 épocas), e o projeto
inteiro ficaria em ~11 h só de treino. O código é o mesmo nos dois lugares
(`pa3/config.py` escolhe o device; em GPU o treino usa precisão mista).

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

## Parte 1 — Trilha C: o professor

**Escolha do professor** (`outputs/part1/01_escolha_professor.png`), feita
**só com os 105 rótulos do orçamento** (que são do train). O val nunca é
mostrado ao professor. O test_public aparece só como referência.

| professor (pesos abertos, rodando local) | F1 nos 105 rótulos (center · ensemble) | F1 zero-shot test_public: só nomes das classes | … ensemble de prompts |
|---|---|---|---|
| CLIP ViT-B/32 (OpenAI) | 0,23 | 0,223 | 0,229 |
| BioCLIP (ViT-B/16) | 0,36 | 0,216 | 0,350 |
| **BioCLIP 2 (ViT-L/14)** | **0,60** | **0,401** | **0,649** |

- A linha de referência obrigatória (nomes das classes como prompt) é **0,401**.
  Com o ensemble de prompts (nome comum + científico, "camera trap photo",
  "at night", "infrared") o mesmo modelo vai a **0,649**, acima do piso (0,460).
- **Center crop venceu letterbox** em todos os professores: a foto inteira com
  faixas cinza deixa o animal ainda menor nos 224 px do ViT.
- **Professor melhorado** com os rótulos do orçamento: protótipo de cada
  classe = `normalize(texto + β · média das imagens rotuladas)`, com β e a
  temperatura escolhidos por leave-one-out nos próprios rótulos.

| orçamento | β (LOO) | professor melhorado, F1 test_public |
|---|---|---|
| 1/classe | 1,0 (padrão: LOO impossível) | 0,671 |
| 5/classe | 0,25 | 0,682 |
| 20/classe | 1,0 | 0,738 |

**Onde o professor erra** (matriz de ruído: `03_matriz_ruido_teacher_zs.png`):

- **Woodchuck é uma classe-ímã.** O professor zero-shot prevê woodchuck 134
  vezes no test_public, onde só há 22 (acerta 13). Essas previsões têm
  confiança média 0,44 (contra 0,67 nas demais), e o texto "woodchuck" é o
  mais próximo do centróide de todas as imagens. Imagem com pouca evidência
  (animal pequeno, borrado) cai na classe mais "genérica". Os maiores
  vazamentos: dog→woodchuck (24), chipmunk→woodchuck (15), gray
  squirrel→woodchuck (15). O professor melhorado reduz os pseudo-rótulos de
  woodchuck no train de 835 para ~320 (`05_vies_distribuicao.png`).
- **Noite não é o problema, é o contrário** (`04_professor_noite_vs_dia.png`).
  48% do test_public é noturno em IR (detectado pela diferença entre canais
  RGB, com separação limpa). O professor acerta 0,73 à noite e 0,54 de dia.
  Parte disso é a mistura de espécies (as diurnas, como passarinhos, esquilos
  e galinha, são mais difíceis), mas o efeito se mantém **dentro da mesma
  classe**: grey_fox 0,00→0,71, bird 0,27→0,86, dog 0,23→0,55. A exceção é
  domestic_cat (0,71→0,32). Hipótese: o flash IR ilumina o animal próximo e
  apaga o fundo. O piso não tem essa vantagem (0,49 dia, 0,48 noite).
- **Classes parecidas:** crow→fox_squirrel (9), cottontail→opossum (8),
  bobcat→coyote (7). bird↔american_crow quase não se confundem no professor
  (0 e 0).
- **Precisão × cobertura** (`06_precisao_cobertura.png`, diagnóstico no
  test_public, não seleção): com τ=0,8 os pseudo-rótulos do professor
  melhorado têm ~98% de precisão em ~45% das imagens. Com τ=0,5, ~88% em ~67%.

Análise por **tamanho do animal** (MegaDetector) e os **alunos** saem do
Colab (ver acima).
