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

```
dataset/
├── images/        # 85 imagens (.jpg/.jpeg)
├── labels/        # 85 arquivos YOLO (.txt) — uma box por imagem
├── classes.txt    # torneira_aberta / torneira_fechada
└── notes.json     # metadados do export do Label Studio
```

## Notebook

[`torneiras_estados_yolo.ipynb`](./torneiras_estados_yolo.ipynb) — pronto para
o **Google Colab** (`Ambiente de execução → Executar tudo`). Ele:

1. Instala o Ultralytics (YOLO11) e checa GPU;
2. Baixa o dataset direto do GitHub (ou usa a pasta local);
3. Explora a distribuição de classes e visualiza as anotações;
4. Faz um **split estratificado** treino/validação (80/20, reprodutível);
5. Treina um **baseline** de 2 épocas (referência do enunciado);
6. Treina o **modelo completo** com transfer learning + data augmentation +
   early-stopping para o melhor resultado possível;
7. Reporta **métricas** (P, R, mAP@50, mAP@50-95), matriz de confusão e curvas;
8. Roda **inferência** na validação e em **imagens novas** (upload no Colab).

### Decisões para maximizar o resultado

- **Modelo:** `yolo11s.pt` (transfer learning a partir do COCO).
- **Augmentation forte:** HSV, rotação, escala, translação, flip horizontal,
  *mosaic* + *mixup* — compensa o tamanho pequeno do dataset.
- **`cos_lr` + `patience`:** decaimento cosseno do learning rate e
  early-stopping para parar no melhor ponto sem overfitting.
- **Split estratificado por classe** e reprodutível (`SEED=42`).

> **Caveat de honestidade científica:** como há várias fotos por integrante, um
> *group split* (separar integrantes entre treino e validação) daria métricas
> mais conservadoras/generalizáveis. Optou-se pelo split estratificado por
> classe (padrão deste desafio); o caveat fica registrado no notebook.
