# Project 2 — Desafio Semana 5 (Visão Computacional)

Detecção de **estado** de um objeto com YOLO — torneira **aberta** vs. **fechada** (Residência em IA, UniSENAI — Prof. Matheus Vanzan).

## Objetivo

Treinar um modelo YOLO capaz de distinguir dois estados de um mesmo objeto (torneira aberta/fechada com base na presença do jato de água), usando um mini-dataset autoral fotografado pelo grupo, anotado e exportado em formato COCO.

## Conexão com a indústria

Detectar se uma torneira está aberta ou fechada por visão computacional é o mesmo problema, em miniatura, que diversas inspeções industriais baseadas em **estado visual de um equipamento**:

- **Monitoramento de válvulas em linhas de processo** (água, vapor, óleo, gás): confirmar visualmente se uma válvula está aberta/fechada quando o sensor de posição não é confiável ou não existe, útil como camada redundante de SCADA/CLP.
- **Detecção de vazamento/fluxo**: a presença do jato de água é análoga à presença de fluido escoando em um ponto que deveria estar seco — usado em inspeção de juntas, drenos e conexões.
- **Gestão hídrica/energética**: identificar torneiras, registros ou válvulas deixados abertos indevidamente (desperdício de água em ambientes industriais, hospitalares ou residenciais), disparando um alarme.
- **Inspeção de painéis e maquinário**: o mesmo princípio (estado A vs. estado B do mesmo objeto, não dois objetos diferentes) se aplica a luzes indicadoras ligadas/apagadas, registros de gás abertos/fechados e portas de segurança.

## Dataset

- **Fonte**: mini-dataset autoral (`torneiras-vc-final`), fotografado pelo grupo e anotado/exportado via [Roboflow](https://roboflow.com) em formato **COCO**.
- **69 imagens**, 1 bounding box por imagem, 2 classes: `aberta` (jato de água visível) e `fechada` (torneira seca).
- **Splits originais do export**: 48 treino / 10 validação / 11 teste.
- **Diversidade**: contribuições de ~10 pessoas (prefixo `s01`...`s10`), cobrindo torneiras de cozinha e banheiro, diferentes modelos (monocomando, gravetes, bica alta/baixa), ângulos, distâncias, fundos e condições de iluminação (natural/artificial, quente/fria).
- O split de **teste** nunca é usado em treino/validação — ele faz o papel das "fotos novas tiradas depois do treinamento" pedidas no enunciado.

Estrutura:
```
data/
  raw/        # dataset original em COCO (train/valid/test + _annotations.coco.json)
  yolo/       # gerado por scripts/coco_to_yolo.py (images/ + labels/ + data.yaml) — não versionado
```

## Metodologia

1. **Conversão COCO → YOLO** (`scripts/coco_to_yolo.py`): remapeia as categorias COCO (descartando a categoria-raiz do projeto Roboflow) para classes YOLO 0-indexed e normaliza as bounding boxes.
2. **Transfer learning**: todos os modelos partem de pesos pré-treinados em COCO (`yolov8n`, `yolov8s`, `yolo11n`) — com só 48 imagens de treino, treinar do zero não é viável.
3. **Grid de experimentos** (`scripts/train.py`), variando exatamente o que o desafio pede ("teste diferentes versões, configurações"):

   | Config | Modelo | Épocas | Resolução | Objetivo |
   |---|---|---|---|---|
   | `baseline_yolov8n_e2` | YOLOv8n | 2 | 640 | baseline obrigatório (poucas épocas) |
   | `yolov8n_e80` | YOLOv8n | 80 (patience=20) | 640 | configuração "padrão" otimizada |
   | `yolov8n_e80_img416` | YOLOv8n | 80 (patience=20) | 416 | efeito de reduzir a resolução |
   | `yolov8s_e80` | YOLOv8s | 80 (patience=20) | 640 | efeito de um modelo maior |
   | `yolo11n_e80` | YOLO11n | 80 (patience=20) | 640 | efeito de uma versão mais nova do YOLO |

   Todos os treinos rodaram em **CPU** (ambiente sem GPU), com `batch=8`, `cache=True` (dataset cabe inteiro em RAM) e `seed=0` fixo para reprodutibilidade.
4. **Avaliação** no split de teste (11 imagens nunca vistas) com métricas padrão de detecção: precisão, recall, mAP50 e mAP50-95.
5. **Inspeção visual** das predições no teste (`scripts/predict_samples.py`) para listar acertos e erros imagem a imagem.

## Resultados

<!-- RESULTS_TABLE -->

## Análise de erros

<!-- ERROR_ANALYSIS -->

## Como rodar

```bash
pip install ultralytics

# 1. Converter o dataset COCO para YOLO
python3 scripts/coco_to_yolo.py --src data/raw --dst data/yolo

# 2. Treinar todas as configurações do grid
python3 scripts/train.py --configs all
# ou uma config especifica:
python3 scripts/train.py --configs yolov8n_e80

# 3. Rodar o melhor modelo nas imagens de teste e gerar o relatorio de acertos/erros
python3 scripts/predict_samples.py --weights results/runs/<melhor_config>/weights/best.pt
```

## Decisões técnicas

- **`device=cpu`**: ambiente de treino sem GPU disponível; o dataset é pequeno o suficiente para isso não ser um problema (cada época de YOLOv8n leva ~10-15s).
- **`cache=True`**: com apenas 69 imagens, cabem inteiras em RAM — evita reler/decodificar JPEG a cada época.
- **Pesos pré-treinados (COCO)** em vez de treino do zero: 48 imagens de treino não são suficientes para aprender features visuais do zero.
- **`patience=20`** nos treinos longos: early stopping para evitar overfitting num dataset tão pequeno, sem precisar fixar manualmente o número ideal de épocas.
- **Split de teste do próprio export** usado como "imagens novas": como o Roboflow já separa um holdout de 11 imagens nunca usadas em treino/validação, ele cumpre o papel de "fotos tiradas depois do treinamento" exigido no enunciado.
- **`batch=8`**: dataset pequeno (48 imagens de treino) — batches maiores não trariam benefício e o `optimizer=auto` do Ultralytics já ajusta o LR para o batch size.
