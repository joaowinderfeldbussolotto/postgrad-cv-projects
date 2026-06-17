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
o **Google Colab** (`Ambiente de execução → Executar tudo`, com **GPU T4**). Ele:

1. Instala o Ultralytics (YOLO11) e checa GPU;
2. Baixa o dataset direto do GitHub (ou usa a pasta local);
3. Explora a distribuição de classes e visualiza as anotações;
4. Treina um **baseline** de 2 épocas (referência do enunciado);
5. Treina com **validação cruzada 5-fold** (resultado principal) e reporta
   **mAP@50 / mAP@50-95 médios ± desvio** — métrica confiável;
6. **Diagnóstico** por matriz de confusão (erro de *localização* vs *estado*);
7. **Compara `yolo11n/s/m`** para mostrar que modelos maiores não compensam;
8. Roda **inferência** na validação e em **imagens novas** (upload no Colab).

### Decisões para maximizar o resultado

- **Modelo:** `yolo11s.pt` (transfer learning a partir do COCO). Modelos XL
  **não** são usados: em 85 imagens overfitam e não generalizam melhor.
- **Augmentation calibrada para o problema:** HSV/flip/rotação/translação **leves**,
  com **`mosaic=0` e `mixup=0` desligados** — eles encolhem a torneira e misturam
  aberta+fechada, destruindo a pista de estado (ótimos para COCO, ruins aqui).
- **`imgsz=768`** para preservar o detalhe do registro/alavanca.
- **`cos_lr` + `patience`:** decaimento cosseno do LR e early-stopping.
- **Validação cruzada 5-fold** estratificada e reprodutível (`SEED=42`): evita a
  métrica ruidosa de um único split de 17 imagens.

> **Caveat de honestidade científica:** como há várias fotos por integrante, um
> *group split* (separar integrantes entre treino e validação) daria métricas
> mais conservadoras/generalizáveis. Optou-se pelo split estratificado por
> classe (padrão deste desafio); o caveat fica registrado no notebook.

### Histórico

- **v1:** treino único com augmentation forte → `mAP50 ≈ 0.38` (melhor época = 4),
  sinal de augmentation agressiva demais para o problema.
- **v2 (atual):** augmentation calibrada + 5-fold CV + `imgsz=768` + diagnóstico +
  comparação de modelos, para um resultado mais alto e **confiável**.
