from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputsFinal"


def load_selected_word() -> str:
    payload = json.loads((OUTPUT_DIR / "selected_word.json").read_text(encoding="utf-8"))
    return payload["selected_word"]


def build_inventory() -> pd.DataFrame:
    rows = []
    description_map = {
        "corpus_overview.csv": "High-level corpus and occurrence summary",
        "occurrences.csv": "All extracted occurrences with context windows",
        "occurrence_examples.csv": "Readable sample of extracted contexts",
        "ner_entity_frequencies.csv": "Named entities aggregated by text and label",
        "ner_cooccurrence_examples.csv": "Examples where the concept appears with named entities",
        "bert_cluster_examples.csv": "Representative examples from each embedding cluster",
        "classifier_labeled_subset.csv": "Balanced weak-label training subset",
        "classifier_metrics.json": "Held-out evaluation metrics for logistic regression",
        "classifier_predictions.csv": "Predictions for all extracted occurrences",
    }

    for path in sorted(OUTPUT_DIR.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_kb": round(path.stat().st_size / 1024, 2),
                "description": description_map.get(path.name, ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    selected_word = load_selected_word()
    corpus_overview = pd.read_csv(OUTPUT_DIR / "corpus_overview.csv").iloc[0]
    distribution = pd.read_csv(OUTPUT_DIR / "document_occurrence_distribution.csv")
    ner_focus = pd.read_csv(OUTPUT_DIR / "ner_focus_entities.csv")
    bert_summary = pd.read_csv(OUTPUT_DIR / "bert_cluster_summary.csv")
    classifier_metrics = json.loads((OUTPUT_DIR / "classifier_metrics.json").read_text(encoding="utf-8"))
    top_features = pd.read_csv(OUTPUT_DIR / "top_features.csv")

    top_docs = distribution.head(5)
    top_entities = ner_focus.head(8)
    top_cluster_lines = [
        f"- Cluster {int(row.cluster_id)} ({row.inferred_theme}): {int(row.cluster_size)} contexts; {row.top_terms}"
        for row in bert_summary.itertuples(index=False)
    ]
    negative_features = top_features.sort_values("coefficient").head(3)
    positive_features = top_features.sort_values("coefficient", ascending=False).head(3)
    feature_lines = [
        *[
            f"- `{row.feature}` -> {row.direction} ({row.coefficient:.3f})"
            for row in negative_features.itertuples(index=False)
        ],
        *[
            f"- `{row.feature}` -> {row.direction} ({row.coefficient:.3f})"
            for row in positive_features.itertuples(index=False)
        ],
    ]

if __name__ == "__main__":
    main()
