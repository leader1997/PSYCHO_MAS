"""Generate SVG charts for scientific_paper_inclusion.md."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

MODEL_AUC = [
    ("XGBoost", 0.853),
    ("LightGBM", 0.852),
    ("CatBoost", 0.852),
    ("FT-Transformer", 0.852),
    ("TabTransformer", 0.851),
    ("DCNv2", 0.850),
    ("Random Forest", 0.849),
    ("Extra Trees", 0.848),
    ("Log. Regression", 0.847),
]

TOP_FEATURES = [
    ("aca_impa", 0.300),
    ("lonely", 0.169),
    ("flourish", 0.139),
    ("fincur", 0.082),
    ("positiveMH", 0.049),
    ("age", 0.033),
    ("food_worry", 0.031),
    ("sleep_wknight", 0.029),
]

RETRIEVAL = [
    ("BM25", "hit@10", 0.29),
    ("Hybrid FAISS", "hit@10", 0.32),
    ("BM25", "mrr", 0.092),
    ("Hybrid FAISS", "mrr", 0.124),
]

COLORS = {"BM25": "#4E79A7", "Hybrid FAISS": "#59A14F"}


def _hbar(path: Path, title: str, items: list[tuple[str, float]], color: str = "#4E79A7") -> None:
    items = sorted(items, key=lambda x: x[1])
    w, h = 720, 40 + len(items) * 36
    ml, pw = 140, 520
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, val) in enumerate(items):
        y = 40 + i * 36
        bw = val * pw
        svg.append(f'<text x="{ml-8}" y="{y+16}" text-anchor="end" font-family="Arial" font-size="12">{label}</text>')
        svg.append(f'<rect x="{ml}" y="{y}" width="{bw:.1f}" height="22" fill="{color}" rx="2"/>')
        svg.append(f'<text x="{ml+bw+6:.0f}" y="{y+16}" font-family="Arial" font-size="11">{val:.3f}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _grouped_retrieval(path: Path) -> None:
    w, h = 520, 320
    ml, pw, ph = 70, 400, 200
    groups = [("hit@10", 0.29, 0.32), ("MRR", 0.092, 0.124)]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="260" y="28" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700">Retrieval: BM25 vs Hybrid (100-Q benchmark)</text>',
    ]
    gw = pw / 2
    for gi, (label, b, hy) in enumerate(groups):
        x0 = ml + gi * gw
        maxv = 0.35 if gi == 0 else 0.15
        for vi, (name, val, col) in enumerate([("BM25", b, COLORS["BM25"]), ("Hybrid", hy, COLORS["Hybrid FAISS"])]):
            x = x0 + 40 + vi * 70
            bh = (val / maxv) * ph
            y = 80 + ph - bh
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="50" height="{bh:.1f}" fill="{col}"/>')
            svg.append(f'<text x="{x+25}" y="{y-4}" text-anchor="middle" font-size="10">{val:.3f}</text>')
        svg.append(f'<text x="{x0+gw/2}" y="300" text-anchor="middle" font-size="12">{label}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    _hbar(CHARTS / "model_auc_comparison.svg", "Test AUC by model (HMS 2024–2025)", MODEL_AUC, "#4E79A7")
    _hbar(CHARTS / "aggregated_feature_importance.svg", "Top aggregated XAI features (9-model consensus)", TOP_FEATURES, "#E15759")
    _grouped_retrieval(CHARTS / "retrieval_bm25_vs_hybrid.svg")
    print(f"Charts written to {CHARTS}")


if __name__ == "__main__":
    main()
