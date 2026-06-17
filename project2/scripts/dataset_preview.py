"""Gera um grid de imagens de exemplo (com bounding boxes) para cada classe,
para ilustrar o dataset no README.

Uso:
    python3 dataset_preview.py
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
YOLO_DIR = HERE / ".." / "data" / "yolo"
OUT_PATH = HERE / ".." / "results" / "artifacts" / "dataset_preview.png"
CLASS_NAMES = ["aberta", "fechada"]
COLORS = {"aberta": (0, 200, 0), "fechada": (0, 0, 220)}
N_PER_CLASS = 3


def draw_boxes(img_path: Path, label_path: Path):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        cls_name = CLASS_NAMES[int(cls)]
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        color = COLORS[cls_name]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 6)
        cv2.putText(img, cls_name, (x1, max(0, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    return img


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    images_dir = YOLO_DIR / "images" / "train"
    labels_dir = YOLO_DIR / "labels" / "train"

    by_class = {name: [] for name in CLASS_NAMES}
    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = label_path.read_text().splitlines()
        if not lines:
            continue
        cls_name = CLASS_NAMES[int(lines[0].split()[0])]
        if len(by_class[cls_name]) < N_PER_CLASS:
            img_path = next(images_dir.glob(f"{label_path.stem}.*"))
            by_class[cls_name].append((img_path, label_path))

    fig, axes = plt.subplots(2, N_PER_CLASS, figsize=(4 * N_PER_CLASS, 9))
    for row, cls_name in enumerate(CLASS_NAMES):
        for col, (img_path, label_path) in enumerate(by_class[cls_name]):
            axes[row, col].imshow(draw_boxes(img_path, label_path))
            axes[row, col].axis("off")
            axes[row, col].set_title(cls_name, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130)
    print(f"Preview salvo em {OUT_PATH}")


if __name__ == "__main__":
    main()
