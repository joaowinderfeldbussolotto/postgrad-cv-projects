"""Converte o dataset 'torneiras-vc-final' (exportado do Roboflow em COCO) para o
formato YOLO (images/labels + data.yaml).

Uso:
    python3 coco_to_yolo.py --src /caminho/para/coco --dst /caminho/para/yolo
"""
import argparse
import json
import shutil
from pathlib import Path

SPLITS = ["train", "valid", "test"]
# Pasta "valid" do Roboflow -> "val" no padrão YOLO/Ultralytics
SPLIT_DIR_NAME = {"train": "train", "valid": "val", "test": "test"}


def convert(src: Path, dst: Path):
    images_root = dst / "images"
    labels_root = dst / "labels"
    for d in SPLIT_DIR_NAME.values():
        (images_root / d).mkdir(parents=True, exist_ok=True)
        (labels_root / d).mkdir(parents=True, exist_ok=True)

    # Classes reais são as categorias com supercategory != "none" (a categoria 0
    # é só o nome do projeto no Roboflow, não é uma classe de verdade).
    class_names = None

    for split in SPLITS:
        ann_path = src / split / "_annotations.coco.json"
        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        leaf_categories = sorted(
            (c for c in coco["categories"] if c["supercategory"] != "none"),
            key=lambda c: c["id"],
        )
        names = [c["name"] for c in leaf_categories]
        if class_names is None:
            class_names = names
        elif class_names != names:
            raise ValueError(f"Classes inconsistentes entre splits: {class_names} != {names} ({split})")

        coco_id_to_yolo_id = {c["id"]: i for i, c in enumerate(leaf_categories)}

        images_by_id = {im["id"]: im for im in coco["images"]}
        anns_by_image = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        out_split = SPLIT_DIR_NAME[split]
        for img_id, im in images_by_id.items():
            src_img = src / split / im["file_name"]
            dst_img = images_root / out_split / im["file_name"]
            shutil.copy2(src_img, dst_img)

            w, h = im["width"], im["height"]
            lines = []
            for ann in anns_by_image.get(img_id, []):
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                cls = coco_id_to_yolo_id[ann["category_id"]]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path = labels_root / out_split / (Path(im["file_name"]).stem + ".txt")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        print(f"[{split}] {len(images_by_id)} imagens, {len(coco['annotations'])} anotacoes -> {out_split}/")

    yaml_path = dst / "data.yaml"
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_path.write_text(
        f"""path: {dst.resolve()}
train: images/train
val: images/val
test: images/test
names:
{names_yaml}
""",
        encoding="utf-8",
    )
    print(f"\ndata.yaml escrito em {yaml_path}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Diretorio com train/valid/test em formato COCO")
    parser.add_argument("--dst", type=Path, required=True, help="Diretorio de saida em formato YOLO")
    args = parser.parse_args()
    convert(args.src, args.dst)
