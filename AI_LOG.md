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
  e suave com professor melhorado. A escolha é pelo val. (Depois ampliada
  com o filtro por confiança; ver "Mudança para o Colab".)

### O que aconteceu ao rodar o professor

- **O resultado mudou a história.** Esperávamos que um foundation model
  zero-shot ficasse perto do piso. CLIP e BioCLIP ficaram abaixo (0,23 e
  0,35), mas o BioCLIP 2 fez 0,649 sem nenhum rótulo. A escolha pelos 105
  rótulos do train produziu o mesmo ranking do teste, o que nos deixou
  confiantes de que a seleção sem val funcionou.
- **Hipótese refutada: letterbox.** Apostávamos que mostrar a foto inteira ao
  ViT (em vez do center crop) ajudaria com animais na borda. Piorou em todos
  os modelos. Mantivemos o teste no resultado porque é uma hipótese
  testada e descartada.
- **Não escolhemos β olhando o teste.** Uma varredura de diagnóstico mostrou
  que β=1 daria 0,718 no test_public, contra 0,682 do β=0,25 escolhido por
  leave-one-out. Decidimos manter o β do LOO: trocar seria selecionar no
  test_public, que é exatamente o que o enunciado proíbe.
- **Bug de métrica que pegamos ao olhar a figura.** A primeira figura de
  dia/noite mostrava F1 ~0,3–0,5 para tudo, abaixo do F1 geral, o que não
  fazia sentido. A causa: o F1 macro com `labels=todas as 21 classes` num
  subconjunto conta como zero as classes que nem aparecem nele (não há gambá
  de dia). Criamos `subset_metrics` (F1 só sobre as classes presentes,
  mais a acurácia) e acrescentamos a comparação **dentro da mesma classe**,
  porque dia e noite têm espécies diferentes. A conclusão final ("o
  professor erra mais de dia") só foi aceita depois desse controle.
- **Log "travado".** Durante ~1h o log do professor ficou vazio. Conferimos
  pela CPU e pelos arquivos de cache que ele estava rodando. A causa era um
  `grep` sem `--line-buffered` no encadeamento, que segura a saída em blocos.

### Mudança para o Colab

- O primeiro aluno levou ~10 min por época em CPU (com a CPU já quente),
  então os 3 alunos levariam ~4 h e o projeto inteiro ~11 h de treino, sem
  margem para repetir nada. Decidimos treinar no Colab (T4).
- Para isso: suporte a GPU com precisão mista no laço de treino (o mesmo
  código roda nos dois), pseudo-rótulos versionados no git (o Colab não
  precisa rodar o BioCLIP 2), checkpoints e resultados gravados no Drive a
  cada época, e um notebook (`colab/run_pa3.ipynb`) que só chama os scripts.
- Com GPU sobrando, acrescentamos a família de **filtro por confiança**
  (τ = 0,3 / 0,5 / 0,7, com τ escolhido no val) às variantes do aluno, e
  subimos as épocas do aluno de 8 para 12.
- **Testamos tudo localmente com dados falsos antes de mandar** (scripts da
  Parte 1 e da Parte 2 de ponta a ponta). Isso pegou um `✓` que o console do
  Windows não imprime, e um caso em que o piso da Parte 0 seria retreinado no
  Colab (o checkpoint não vai para o git) e mudaria o número já reportado.
  Agora uma rodada é recarregada só pelo json + logits versionados.
- **MegaDetector para medir o tamanho dos animais.** O enunciado pede para
  avaliar o professor em "animais pequenos", mas o pacote não traz caixas. O
  MegaDetector é permitido na Trilha C e aqui entra só como instrumento de
  medida (train e test_public, nunca val). No Windows o OpenCV que o
  `ultralytics` usa não carrega (falta uma DLL do sistema), então ele roda no
  Colab. Para não depender do pacote PytorchWildlife, que traz muitas
  dependências, pegamos a URL oficial dos pesos lendo o código-fonte dele.

### Depois do Colab: análise dos alunos, da curva e da Parte 3

- **O coelho investigado antes de ser explicado.** O aluno perdeu −0,67 de F1
  em eastern_cottontail em relação ao professor. A primeira hipótese da IA
  (pseudo-rótulos ruins) foi **refutada pelos dados**: os vizinhos dos
  coelhos do teste no espaço do BioCLIP eram coelhos com alta confiança. Só
  depois de olhar as imagens apareceu a causa real: os 26 coelhos são uma
  sequência de uma única câmera, e essa câmera aparece no train com gatos.
  Isso foi confirmado por vizinhos em pixels. A explicação no README é a
  confirmada, não a primeira.
- **Center crop do professor.** Um número levantado na análise, e não
  previsto: 15% do teste tem o animal majoritariamente fora do quadrado
  central que o ViT vê. Ali o professor acerta ~25% e o aluno 71%. Virou um
  painel da figura 11 e a explicação principal de "onde o aluno supera o
  professor".
- **Aluno b20 abaixo do piso b20.** O histórico mostrou que o val escolheu a
  época 2 por um pico isolado. Decidimos **não** reescolher a época olhando o
  teste e registrar como limitação, com a correção que testaríamos (seleção
  de época suavizada).
- **Pesos do modelo final.** A célula que copiava o checkpoint não rodou no
  Colab, e o arquivo foi baixado do Drive à mão. O `predict.py` com os pesos
  exportados reproduz exatamente o F1 do treino (0,6832), conferido antes de
  qualquer texto.
- **Galeria da Parte 3 com seleção por regra.** As 3 imagens de erro e as 5
  de acerto saem de regras escritas antes de ver os mapas, para não escolher
  os exemplos que confirmam a narrativa. Dois dos cinco acertos ("certos
  pelo motivo errado") vieram dessa regra, não de escolha.
- **Revisão de figuras.** Toda figura foi aberta e olhada depois de gerada.
  Foram corrigidos rótulos sobrepostos (curva de orçamento, variantes do
  aluno, tamanho do animal, Parte 3) e uma cor enganosa (perda de −0,40
  pintada como "sem diferença").
