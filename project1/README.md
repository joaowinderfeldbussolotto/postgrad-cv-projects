# Project 1 — Desafio Semana 1 (Visão Computacional)

Notebook do Desafio 1 da disciplina de Visão Computacional (Residência em IA, UniSENAI — Prof. Matheus Vanzan).

## Conteúdo

- `eurosat_cnn_filters.ipynb` — notebook Colab-ready com CNN treinada no dataset **EuroSAT/RGB** (10 classes, 27 000 imagens 64×64), comparando 5 experimentos: baseline RGB (5 e 10 épocas), Grayscale, Canny e Sharpen.

## Como rodar

Abrir no Google Colab e executar as células em ordem. As únicas dependências externas são `tensorflow`, `tensorflow-datasets` e `opencv-python` (já disponíveis no Colab).

## Decisões técnicas

- **Dataset:** EuroSAT/RGB via `tfds.load`. Cor é discriminante em imagens de satélite, o que torna a comparação com filtros que descartam cor (Grayscale, Canny) particularmente informativa.
- **Split 70/15/15** via subsplits do TFDS (`train[:70%]`, etc.) em vez de carregar 27k imagens em memória.
- **Pipeline reutilizável** `build_dataset_with_filter(ds, filter_fn, out_channels)` — o filtro OpenCV é injetado como parâmetro, mantendo o resto do pipeline idêntico entre experimentos.
- **`tf.py_function`** para integrar OpenCV no `tf.data`, sempre seguido de `set_shape` para restaurar a dimensionalidade estática (caso contrário o `model.fit` quebra).
- **CNN única** (3 blocos Conv+Pool → Dense) compartilhada por todos os experimentos — única variável que muda é filtro/épocas.
- **`sparse_categorical_crossentropy`** porque o TFDS já retorna labels inteiros.
- **GaussianBlur 3×3** antes do Canny (e não 5×5 da aula) porque a imagem tem só 64×64.
