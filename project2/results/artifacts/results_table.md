| Config | Modelo | Épocas | Resolução | Precisão | Recall | mAP50 | mAP50-95 | Tempo treino |
|---|---|---|---|---|---|---|---|---|
| `yolov8n_e80` | yolov8n.pt | 80 | 640 | 0.652 | 0.646 | 0.687 | 0.533 | 751s |
| `yolo11n_e80` | yolo11n.pt | 80 | 640 | 0.003 | 0.938 | 0.558 | 0.384 | 285s |
| `yolov8s_e80` | yolov8s.pt | 80 | 640 | 0.461 | 0.633 | 0.471 | 0.332 | 784s |
| `yolov8n_e80_img416` | yolov8n.pt | 80 | 416 | 0.003 | 0.938 | 0.458 | 0.303 | 134s |
| `baseline_yolov8n_e2` | yolov8n.pt | 2 | 640 | 0.003 | 0.938 | 0.328 | 0.221 | 32s |