from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import spacy

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputsFinal"
FIGURES_DIR = OUTPUT_DIR / "figures"

FOCUS_ENTITY_MAP = {
    "authority": ["ORG", "PERSON", "GPE"],
    "commerce": ["GPE", "LOC", "ORG"],
    "credit": ["PERSON", "ORG", "GPE"],
    "justice": ["PERSON", "ORG", "GPE"],
    "law": ["ORG", "PERSON", "GPE"],
    "liberty": ["ORG", "PERSON", "GPE"],
    "market": ["GPE", "LOC", "ORG"],
    "power": ["PERSON", "ORG", "GPE"],
    "state": ["GPE", "ORG", "PERSON"],
    "trade": ["GPE", "LOC", "ORG"],
    "wealth": ["PERSON", "ORG", "GPE"],
}


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_selected_word() -> str:
    payload = json.loads((OUTPUT_DIR / "selected_word.json").read_text(encoding="utf-8"))
    return payload["selected_word"]


def plot_entity_outputs(type_counts: pd.DataFrame, focus_counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_types = type_counts.head(10).sort_values("count")
    axes[0].barh(top_types["entity_label"], top_types["count"], color="#E15759")
    axes[0].set_title("Entity labels")
    axes[0].set_xlabel("Count")

    top_focus = focus_counts.head(12).sort_values("count")
    axes[1].barh(top_focus["entity_text"], top_focus["count"], color="#F28E2B")
    axes[1].set_title("Top focus entities")
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "entity_distribution.png", dpi=200)
    plt.close()


def build_notes(selected_word: str, focus_labels: list[str], type_counts: pd.DataFrame, focus_counts: pd.DataFrame) -> str:
    label_lines = [
        f"- `{row.entity_label}`: {int(row.count)}"
        for row in type_counts.head(8).itertuples(index=False)
    ]
    focus_lines = [
        f"- `{row.entity_text}` ({row.entity_label}): {int(row.count)}"
        for row in focus_counts.head(10).itertuples(index=False)
    ]

    return "\n".join(
        [
            "# NER Notes",
            "",
            f"Selected concept: **{selected_word}**",
            f"Focus entity labels: {', '.join(focus_labels)}",
            "",
            "Overall entity labels:",
            *label_lines,
            "",
            "Most common focus entities:",
            *focus_lines,
            "",
            "Interpretive take:",
            f"- The chosen focus labels match the most useful context around `{selected_word}` in this corpus, especially where the concept is tied to places, institutions, or named actors.",
            "- Historical spelling and OCR noise do limit spaCy's recall, so the tables are best treated as directional evidence rather than a complete census of named entities.",
        ]
    )


def main() -> None:
    ensure_output_dirs()
    selected_word = load_selected_word()
    focus_labels = FOCUS_ENTITY_MAP.get(selected_word, ["PERSON", "ORG", "GPE"])

    occurrences = pd.read_csv(OUTPUT_DIR / "occurrences.csv")
    contexts = occurrences[["context_id", "source_file", "context"]].drop_duplicates("context_id").reset_index(drop=True)

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    entity_rows = []

    for row, doc in zip(contexts.itertuples(index=False), nlp.pipe(contexts["context"].tolist(), batch_size=32)):
        for ent in doc.ents:
            entity_text = " ".join(ent.text.split())
            if len(entity_text) <= 1:
                continue
            if entity_text.lower() == selected_word.lower():
                continue
            if entity_text.islower():
                entity_text = entity_text.title()
            entity_rows.append(
                {
                    "context_id": row.context_id,
                    "source_file": row.source_file,
                    "entity_text": entity_text,
                    "entity_label": ent.label_,
                }
            )

    entity_frame = pd.DataFrame(entity_rows)
    if entity_frame.empty:
        raise ValueError("spaCy did not return any entities for the prepared contexts.")

    type_counts = (
        entity_frame.groupby("entity_label")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    entity_counts = (
        entity_frame.groupby(["entity_label", "entity_text"])
        .agg(count=("context_id", "size"), context_count=("context_id", "nunique"))
        .reset_index()
        .sort_values(["count", "context_count"], ascending=[False, False])
    )
    focus_counts = entity_counts[entity_counts["entity_label"].isin(focus_labels)].copy()

    cooccurrence_examples = (
        entity_frame[entity_frame["entity_label"].isin(focus_labels)]
        .merge(occurrences[["context_id", "selected_word", "source_file", "context"]].drop_duplicates("context_id"), on=["context_id", "source_file"])
        .sort_values(["entity_label", "entity_text", "source_file"])
    )
    cooccurrence_examples = cooccurrence_examples.groupby(["entity_label", "entity_text"]).head(3)

    entity_frame.to_csv(OUTPUT_DIR / "ner_entities_long.csv", index=False)
    type_counts.to_csv(OUTPUT_DIR / "ner_entity_type_frequencies.csv", index=False)
    entity_counts.to_csv(OUTPUT_DIR / "ner_entity_frequencies.csv", index=False)
    focus_counts.to_csv(OUTPUT_DIR / "ner_focus_entities.csv", index=False)
    cooccurrence_examples.to_csv(OUTPUT_DIR / "ner_cooccurrence_examples.csv", index=False)
    plot_entity_outputs(type_counts, focus_counts)

    notes = build_notes(selected_word, focus_labels, type_counts, focus_counts)
    (OUTPUT_DIR / "ner_notes.md").write_text(notes, encoding="utf-8")

    print(f"Ran spaCy NER on {len(contexts)} unique contexts.")
    print(f"Saved focus entity table to {OUTPUT_DIR / 'ner_focus_entities.csv'}")


if __name__ == "__main__":
    main()
