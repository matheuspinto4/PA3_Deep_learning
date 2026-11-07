# AI_LOG

Registro de como usamos IA (Claude Code) ao longo do PA3. Atualizado
conforme avançamos pelas partes do enunciado.

Nenhuma IA foi usada para **rotular ou classificar imagens** (proibido pelo
enunciado). O único rótulo humano usado é o do pacote; os pseudo-rótulos vêm
de um *foundation model* aberto (BioCLIP/CLIP) rodando localmente, como a
Trilha C permite.

## Como trabalhamos

Mesmo esquema do PA2: discutimos a abordagem, nós tomamos as decisões que o
enunciado deixa para a dupla (trilha, backbone, onde rodar), o Claude Code
escreve o código, e nós revisamos cada arquivo, rodamos os scripts e pedimos
ajuste quando um resultado não fazia sentido ou não entendíamos uma escolha.

## Decisões iniciais

- **Trilha C (professor e aluno).** Não temos GPU (PyTorch CPU-only,
  i5-1240P, 16 GB). Medimos antes de decidir: a ResNet-18 treina a ~24 img/s
  em 224×224 nesta máquina. As trilhas A (SimCLR/SwAV, duas vezes) e B
  (FixMatch, muitas épocas com aumentação forte em 6,5k imagens) ficariam
  inviáveis em CPU. Na C, o professor só faz inferência, uma vez.
- **Backbone ResNet-18** (11,19M parâmetros com a cabeça de 21 classes, abaixo
  do limite de 12M). É o mesmo backbone do teto do monitor (0,686), então a
  comparação com o teto é direta.
- **Entrada 224×320, quadro inteiro.** Pedimos para não usar o center crop
  quadrado padrão: num quarto das imagens o animal ocupa <2% do quadro e pode
  estar na borda, e o crop o jogaria fora. 320/224 ≈ 1,43 é a razão de
  aspecto típica das fotos (512×367, 512×359).
- **Cache de imagens.** Decodificar os JPEGs a cada época custaria mais que o
  forward em CPU, então cada split é redimensionado uma vez para `cache/`.

## Parte 0 — o piso

- Uma única receita por arquitetura (`pa3/floor.py`), reutilizada sem mudança
  na Parte 2 para os orçamentos 1/5/20 e as 3 seeds. Assim a curva de
  orçamento compara o mesmo procedimento.
- Época escolhida pelo **F1 macro no val** (empate decidido pela perda de val).
  O test_public só é avaliado depois, com o melhor estado do val.
- Para responder "em que época começa a decorar", pedimos para medir também o
  treino **sem aumentação** a cada época. A perda com aumentação não mostra
  quando a rede já acerta todas as 105 imagens. Os dois marcadores objetivos
  são a época em que a acurácia de treino passa de 99% e a época de menor
  perda no val.
- Aumentação pensada para câmera trap: crop leve (≥70% da área, para não
  tirar o animal pequeno do quadro) e escala de cinza aleatória (p=0,2,
  porque metade das fotos é noturna em IR).

## Parte 1 — Trilha C

- **Quanto do teste é noturno?** Antes de analisar o professor, medimos a
  "colorfulness" (diferença média entre canais RGB) das imagens. A
  distribuição é bimodal com um vão limpo: 391 das 811 imagens do test_public
  têm colorfulness < 1 (IR, cinza puro), e nenhuma está entre 1 e 5. O limiar
  2,0 separa noite de dia sem ambiguidade.
- **Escolha do professor sem tocar no val.** O enunciado proíbe o professor de
  ver o val. Escolhemos modelo (CLIP ViT-B/32, BioCLIP, BioCLIP 2), vista da
  imagem (center crop padrão vs. letterbox com a foto inteira) e prompts
  usando só os 105 rótulos do orçamento, que são do train.
- **Professor melhorado com os 105 rótulos.** O protótipo de cada classe vira
  `normalize(texto + β · média das imagens rotuladas)`. β e a temperatura
  (para os rótulos suaves) são escolhidos por leave-one-out nos próprios
  rótulos do orçamento. Validamos a matemática num caso sintético antes de
  rodar: com texto e imagem desalinhados, o zero-shot acerta 57% e o
  protótipo misto chega a 96% no leave-one-out.
- **Tratamento do ruído como ablação.** Três alunos que diferem em uma coisa
  por vez: duro com zero-shot (sem tratamento), duro com professor melhorado,
  e suave com professor melhorado. A escolha é pelo val.
