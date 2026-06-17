"""Le results/summary.json, gera uma tabela markdown comparando as configs,
um grafico de barras comparando mAP50/mAP50-95, e copia os artefatos do
melhor modelo para results/artifacts/.

Uso:
    python3 make_report.py
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = (HERE / ".." / "results").resolve()
RUNS_DIR = RESULTS_DIR / "runs"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"


def main():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    summary = sorted(summary, key=lambda r: -r["test_map50"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tabela markdown
    lines = [
        "| Config | Modelo | Épocas | Resolução | Precisão | Recall | mAP50 | mAP50-95 | Tempo treino |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| `{r['name']}` | {r['model']} | {r['epochs']} | {r['imgsz']} | "
            f"{r['test_precision']:.3f} | {r['test_recall']:.3f} | {r['test_map50']:.3f} | "
            f"{r['test_map50_95']:.3f} | {r['train_time_s']:.0f}s |"
        )
    table_md = "\n".join(lines)
    (ARTIFACTS_DIR / "results_table.md").write_text(table_md)
    print(table_md)

    # Grafico comparativo
    names = [r["name"] for r in summary]
    map50 = [r["test_map50"] for r in summary]
    map5095 = [r["test_map50_95"] for r in summary]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], map50, width, label="mAP50")
    ax.bar([i + width / 2 for i in x], map5095, width, label="mAP50-95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("mAP (split de teste)")
    ax.set_title("Comparação de configurações — torneiras aberta/fechada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "configs_comparison.png", dpi=150)
    print(f"\nGrafico salvo em {ARTIFACTS_DIR / 'configs_comparison.png'}")

    # Copia artefatos do melhor modelo
    best = summary[0]
    best_run_dir = RUNS_DIR / best["name"]
    for fname in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"]:
        src = best_run_dir / fname
        if src.exists():
            shutil.copy2(src, ARTIFACTS_DIR / f"best_{fname}")

    best_weights = Path(best["weights"])
    if best_weights.exists():
        shutil.copy2(best_weights, ARTIFACTS_DIR / "best.pt")
        print(f"Pesos do melhor modelo ({best['name']}) copiados para {ARTIFACTS_DIR / 'best.pt'}")

    print(f"\nMelhor config: {best['name']} (mAP50={best['test_map50']:.3f})")


if __name__ == "__main__":
    main()
