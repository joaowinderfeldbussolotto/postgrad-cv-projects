"""Roda o melhor modelo treinado nas imagens de teste (nunca vistas em treino/val)
e salva as predicoes anotadas + um relatorio de acertos/erros comparando com o
ground truth.

Uso:
    python3 predict_samples.py --weights ../results/runs/<nome_do_run>/weights/best.pt
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ["aberta", "fechada"]


def load_ground_truth(labels_dir: Path) -> dict:
    gt = {}
    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        classes = [CLASS_NAMES[int(line.split()[0])] for line in lines if line.strip()]
        gt[label_file.stem] = classes
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=HERE / ".." / "data" / "yolo" / "images" / "test")
    parser.add_argument("--labels", type=Path, default=HERE / ".." / "data" / "yolo" / "labels" / "test")
    parser.add_argument("--out", type=Path, default=HERE / ".." / "results" / "artifacts" / "predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    gt_by_image = load_ground_truth(args.labels)

    report = []
    image_paths = sorted(args.images.glob("*"))
    results = model.predict(source=[str(p) for p in image_paths], conf=args.conf, save=False, verbose=False)

    for img_path, result in zip(image_paths, results):
        stem = img_path.stem
        pred_classes = [CLASS_NAMES[int(c)] for c in result.boxes.cls.tolist()] if len(result.boxes) else []
        pred_confs = [round(c, 3) for c in result.boxes.conf.tolist()] if len(result.boxes) else []
        gt_classes = gt_by_image.get(stem, [])

        correct = sorted(pred_classes) == sorted(gt_classes) and len(pred_classes) == len(gt_classes)
        out_path = args.out / f"{stem}_pred.jpg"
        result.save(filename=str(out_path))

        report.append({
            "image": img_path.name,
            "ground_truth": gt_classes,
            "predicted": pred_classes,
            "confidences": pred_confs,
            "correct": correct,
        })
        status = "OK" if correct else "ERRO"
        print(f"[{status}] {img_path.name}: gt={gt_classes} pred={pred_classes} conf={pred_confs}")

    n_correct = sum(r["correct"] for r in report)
    print(f"\n{n_correct}/{len(report)} imagens corretas ({100 * n_correct / len(report):.0f}%)")

    report_path = args.out.parent / "test_predictions_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Relatorio salvo em {report_path}")


if __name__ == "__main__":
    main()
