"""Treina e avalia varias configuracoes de YOLO no dataset de torneiras (aberta/fechada).

Uso:
    python3 train.py --data ../data/yolo/data.yaml --project ../results/runs --configs all
    python3 train.py --configs baseline_yolov8n_e2
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

# Cada experimento varia modelo (versao/tamanho), resolucao e epocas para
# comparar o efeito de cada configuracao no dataset pequeno (69 imagens).
CONFIGS = [
    {
        "name": "baseline_yolov8n_e2",
        "model": "yolov8n.pt",
        "epochs": 2,
        "imgsz": 640,
        "patience": 0,
    },
    {
        "name": "yolov8n_e80",
        "model": "yolov8n.pt",
        "epochs": 80,
        "imgsz": 640,
        "patience": 20,
    },
    {
        "name": "yolov8n_e80_img416",
        "model": "yolov8n.pt",
        "epochs": 80,
        "imgsz": 416,
        "patience": 20,
    },
    {
        "name": "yolov8s_e80",
        "model": "yolov8s.pt",
        "epochs": 80,
        "imgsz": 640,
        "patience": 20,
    },
    {
        "name": "yolo11n_e80",
        "model": "yolo11n.pt",
        "epochs": 80,
        "imgsz": 640,
        "patience": 20,
    },
]

COMMON_KWARGS = dict(
    batch=8,
    device="cpu",
    seed=0,
    cache=True,
    workers=4,
    verbose=False,
    plots=True,
    val=True,
)


def run_config(cfg: dict, data_yaml: Path, project_dir: Path) -> dict:
    print(f"\n{'=' * 70}\n>>> Treinando: {cfg['name']}  ({cfg['model']}, epochs={cfg['epochs']}, imgsz={cfg['imgsz']})\n{'=' * 70}")
    model = YOLO(cfg["model"])

    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        patience=cfg["patience"],
        project=str(project_dir),
        name=cfg["name"],
        exist_ok=True,
        **COMMON_KWARGS,
    )
    train_time = time.time() - t0

    # Avalia no split de teste (imagens nunca vistas em treino/val)
    metrics = model.val(data=str(data_yaml), split="test", project=str(project_dir), name=f"{cfg['name']}_test", exist_ok=True)

    result = {
        "name": cfg["name"],
        "model": cfg["model"],
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "train_time_s": round(train_time, 1),
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
        "test_map50": float(metrics.box.map50),
        "test_map50_95": float(metrics.box.map),
        "weights": str(project_dir / cfg["name"] / "weights" / "best.pt"),
    }
    print(f">>> {cfg['name']}: P={result['test_precision']:.3f} R={result['test_recall']:.3f} "
          f"mAP50={result['test_map50']:.3f} mAP50-95={result['test_map50_95']:.3f} "
          f"tempo={train_time:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE / ".." / "data" / "yolo" / "data.yaml")
    parser.add_argument("--project", type=Path, default=HERE / ".." / "results" / "runs")
    parser.add_argument("--configs", nargs="+", default=["all"], help="nomes dos configs ou 'all'")
    parser.add_argument("--summary-out", type=Path, default=HERE / ".." / "results" / "summary.json")
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    project_dir = args.project.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    if args.configs == ["all"]:
        selected = CONFIGS
    else:
        selected = [c for c in CONFIGS if c["name"] in args.configs]

    summary_path = args.summary_out.resolve()
    results = []
    if summary_path.exists():
        results = json.loads(summary_path.read_text())

    for cfg in selected:
        result = run_config(cfg, data_yaml, project_dir)
        results = [r for r in results if r["name"] != cfg["name"]] + [result]
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n\n=== RESUMO FINAL ===")
    for r in sorted(results, key=lambda r: -r["test_map50"]):
        print(f"{r['name']:28s} mAP50={r['test_map50']:.3f}  mAP50-95={r['test_map50_95']:.3f}  "
              f"P={r['test_precision']:.3f}  R={r['test_recall']:.3f}  tempo={r['train_time_s']:.0f}s")


if __name__ == "__main__":
    main()
