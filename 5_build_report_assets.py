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
        "selected_word_summary.md": "Concept recommendation and shortlist explanation",
        "corpus_overview.csv": "High-level corpus and occurrence summary",
        "occurrences.csv": "All extracted occurrences with context windows",
        "occurrence_examples.csv": "Readable sample of extracted contexts",
        "ner_entity_frequencies.csv": "Named entities aggregated by text and label",
        "ner_cooccurrence_examples.csv": "Examples where the concept appears with named entities",
        "bert_cluster_examples.csv": "Representative examples from each embedding cluster",
        "classifier_labeled_subset.csv": "Balanced weak-label training subset",
        "classifier_metrics.json": "Held-out evaluation metrics for logistic regression",
        "classifier_predictions.csv": "Predictions for all extracted occurrences",
        "interpretive_notes.md": "Cross-method notes for paper drafting",
        "project_overview.md": "Guide to the most important outputs",
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

    interpretive_notes = "\n".join(
        [
            "# Interpretive Notes",
            "",
            f"Selected concept: **{selected_word}**",
            "",
            "Working claim direction:",
            f"- `{selected_word}` is not used in a single stable way across this corpus. The saved outputs suggest a structured split between at least two recurring uses, supported by both contextual clustering and the classifier.",
            "",
            "Core descriptive facts:",
            f"- Total occurrences: {int(corpus_overview['total_occurrences'])}",
            f"- Documents containing the word: {int(corpus_overview['documents_with_word'])}",
            f"- Held-out classifier accuracy: {classifier_metrics['accuracy']:.3f}",
            f"- Held-out classifier macro F1: {classifier_metrics['macro_f1']:.3f}",
            "",
            "Documents to inspect first:",
            *[
                f"- `{row.source_file}`: {int(row.occurrence_count)} occurrences"
                for row in top_docs.itertuples(index=False)
            ],
            "",
            "NER takeaways:",
            *[
                f"- `{row.entity_text}` ({row.entity_label}) appears {int(row.count)} times in the focus-entity table."
                for row in top_entities.itertuples(index=False)
            ],
            "- OCR noise and early modern spelling clearly limit off-the-shelf NER, so the named-entity results are best used as strong examples and frequency signals rather than a complete ground truth.",
            "",
            "BERT cluster takeaways:",
            *top_cluster_lines,
            "",
            "Classifier takeaways:",
            *feature_lines,
            "- The labels are weakly inferred, so the classifier is most valuable as an interpretive aid: it shows which surrounding words separate the two uses and how broadly those uses distribute across the corpus.",
            "",
            "Limitations to mention in the paper:",
            "- OCR normalization and historical spelling create noisy sentence boundaries and imperfect named-entity extraction.",
            "- Weak labels introduce some circularity, even though the direct rule keywords were masked before model fitting.",
            "- Clusters are heuristic groupings of contexts, so they still need close reading rather than being treated as final meanings.",
        ]
    )
    (OUTPUT_DIR / "interpretive_notes.md").write_text(interpretive_notes, encoding="utf-8")

    overview = "\n".join(
        [
            "# Project Overview",
            "",
            f"This run is organized around the concept word **{selected_word}**.",
            "",
            "Open these files first:",
            "- `outputsFinal/selected_word_summary.md` for the concept choice and shortlist.",
            "- `outputsFinal/occurrence_examples.csv` for readable context windows.",
            "- `outputsFinal/ner_focus_entities.csv` and `outputsFinal/figures/entity_distribution.png` for named-entity patterns.",
            "- `outputsFinal/bert_cluster_examples.csv` and `outputsFinal/figures/bert_clusters.png` for contextual variation.",
            "- `outputsFinal/classifier_metrics.json`, `outputsFinal/top_features.csv`, and `outputsFinal/figures/classifier_confusion_matrix.png` for the supervised model.",
            "- `outputsFinal/interpretive_notes.md` for a paper-writing oriented synthesis.",
            "",
            "If you need a full artifact list, check `outputsFinal/artifact_inventory.csv`.",
        ]
    )
    (OUTPUT_DIR / "project_overview.md").write_text(overview, encoding="utf-8")

    inventory = build_inventory()
    inventory.to_csv(OUTPUT_DIR / "artifact_inventory.csv", index=False)

    print(f"Built consolidated report assets for '{selected_word}'.")
    print(f"Saved overview to {OUTPUT_DIR / 'project_overview.md'}")


if __name__ == "__main__":
    main()
