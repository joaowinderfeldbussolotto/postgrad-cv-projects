# Project 5 — Detecção de Estados de Torneiras com YOLO 🚰

**Unidade Curricular:** Visão Computacional — Residência em Inteligência Artificial
**Desafio Semana 5:** Detecção de Estados com YOLO

## Objetivo

Treinar um modelo **YOLO** capaz de distinguir o **estado** de uma torneira a
partir de fotos próprias do grupo:

| Classe | id | Descrição |
|--------|----|-----------|
| `torneira_aberta`  | 0 | Torneira aberta (em uso) |
| `torneira_fechada` | 1 | Torneira fechada (desligada) |

## 🏭 Conexão com a indústria

Detectar **aberto/fechado** de uma torneira é equivalente ao **monitoramento de
válvulas e registros industriais** (tubulações, plantas químicas, sistemas de
utilidades). Identificar automaticamente o estado de um atuador previne
vazamentos, desperdício e falhas operacionais.

## Dataset

- **85 imagens** anotadas no **Label Studio** (formato YOLO), exportadas para
  `dataset/`.
- Variações de iluminação, ângulo, distância e fundo.
- Uma bounding box por imagem.
- Distribuição: `torneira_aberta` = 36 boxes · `torneira_fechada` = 49 boxes.
- Fotos de 10 integrantes (prefixo `s01`…`s10` no nome do arquivo).
- ⚠️ **24 imagens foram exportadas em duplicidade** (mesma foto, anotada duas
  vezes — 56% do dataset). O notebook detecta essas duplicatas por hash do
  arquivo e usa split *group-aware* para que nunca fiquem divididas entre
  treino e validação (ver Seção 5/6 do notebook).

```
dataset/
├── images/        # 85 imagens (.jpg/.jpeg)
├── labels/        # 85 arquivos YOLO (.txt) — uma box por imagem
├── classes.txt    # torneira_aberta / torneira_fechada
└── notes.json     # metadados do export do Label Studio
```

## Notebooks

Há **dois notebooks**, ambos prontos para o **Google Colab**
(`Ambiente de execução → Executar tudo`, com **GPU T4**). A diferença entre eles
é só **como o split treino/validação é agrupado**:

| Notebook | Agrupamento do split | Métrica |
|----------|----------------------|---------|
| [`torneiras_estados_yolo.ipynb`](./torneiras_estados_yolo.ipynb) | por **hash de imagem** (duplicatas exatas nunca cruzam) | mais alta, mas fotos diferentes da mesma torneira podem aparecer dos dois lados |
| [`torneiras_estados_yolo_split_integrante.ipynb`](./torneiras_estados_yolo_split_integrante.ipynb) | por **integrante** (`sNN` — cada pessoa/torneira 100% de um lado) | mais baixa/variável, porém mede generalização real para uma torneira **nunca vista** |

Os dois clonam o dataset direto da branch `claude/wonderful-fermi-mwei17` do
GitHub (onde o `project5/dataset` existe — ainda não está no `main`) e, na
inferência, usam `agnostic_nms=True, max_det=1`: como uma torneira não pode
estar aberta **e** fechada ao mesmo tempo, em caso de duas detecções
conflitantes o modelo mantém só a de **maior confiança**.

[`torneiras_estados_yolo.ipynb`](./torneiras_estados_yolo.ipynb) — notebook
principal (split por hash). Ele:

1. Instala o Ultralytics (YOLO11) e checa GPU;
2. Baixa o dataset direto do GitHub (ou usa a pasta local);
3. Explora a distribuição de classes e visualiza as anotações;
4. **Detecta imagens duplicadas** por hash e usa split *group-aware* (evita
   vazamento de dados entre treino/validação);
5. Treina um **baseline** de 2 épocas (referência do enunciado);
6. Treina com **validação cruzada 5-fold** (resultado principal, group-aware)
   e reporta **mAP@50 / mAP@50-95 / P / R médios ± desvio** — métrica confiável;
7. **Prova** (verificação independente) que nenhuma imagem idêntica está em
   treino e validação ao mesmo tempo, no baseline e em todos os folds;
8. **Diagnóstico** por matriz de confusão (erro de *localização* vs *estado*);
9. Roda **inferência** na validação e em **imagens novas** (upload no Colab).

> ⚠️ **Sobre a variância entre folds:** mesmo com a prova de que nenhuma imagem
> idêntica vaza, os folds variam (~0.4 a ~0.9 de mAP50) porque há **poucas cenas
> independentes** (10 integrantes; s04/s05/s06 = 62%) — fotos diferentes da mesma
> torneira deixam o modelo memorizar a cena. Leia a métrica pela **média ±
> desvio**; o caminho para subir/estabilizar é **mais torneiras independentes**.
> O notebook [`_split_integrante`](./torneiras_estados_yolo_split_integrante.ipynb)
> resolve isso na raiz, agrupando por integrante.

[`torneiras_estados_yolo_split_integrante.ipynb`](./torneiras_estados_yolo_split_integrante.ipynb)
— mesmo pipeline, mas o split (baseline + CV) agrupa por **integrante** (`sNN`),
não por hash: cada pessoa/torneira fica 100% de um lado, então cada fold valida
numa torneira **nunca vista** durante o treino. A prova da Seção 9 verifica essa
garantia em dobro (integrante **e** hash). Resultado: métrica mais
baixa/variável, mas mais honesta sobre generalização real.

### Decisões para maximizar o resultado

- **Modelo:** `yolo11s.pt` (transfer learning a partir do COCO). Modelos maiores
  (`m`, `l`) foram testados e overfitaram — sem ganho de generalização e com
  treino mais lento — por isso `yolo11s` é a escolha final.
- **Augmentation calibrada para o problema:** HSV/flip/rotação/translação **leves**,
  com **`mosaic=0` e `mixup=0` desligados** — eles encolhem a torneira e misturam
  aberta+fechada, destruindo a pista de estado (ótimos para COCO, ruins aqui).
- **`imgsz=768`** para preservar o detalhe do registro/alavanca.
- **`cos_lr` + `patience`:** decaimento cosseno do LR e early-stopping.
- **Validação cruzada 5-fold *group-aware*** (`StratifiedGroupKFold`, `SEED=42`):
  evita a métrica ruidosa de um único split pequeno **e** evita que imagens
  duplicadas vazem entre treino e validação.

> **Caveat de honestidade científica:** este notebook usa split por hash (padrão
> deste desafio, com a correção de leak). O notebook
> [`_split_integrante`](./torneiras_estados_yolo_split_integrante.ipynb) implementa
> o *group split* por integrante sugerido aqui, com métricas mais
> conservadoras/generalizáveis.
