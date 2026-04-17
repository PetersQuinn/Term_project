from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
TEXTS_DIR = PROJECT_ROOT / "texts"
OUTPUT_DIR = PROJECT_ROOT / "outputsFinal"
FIGURES_DIR = OUTPUT_DIR / "figures"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_selected_word() -> str:
    payload = json.loads((OUTPUT_DIR / "selected_word.json").read_text(encoding="utf-8"))
    return payload["selected_word"]


def split_sentences(text: str) -> list[str]:
    sentences = []
    for part in SENTENCE_SPLIT_RE.split(text.replace("\r", "\n")):
        cleaned = " ".join(part.split())
        if not cleaned:
            continue
        words = cleaned.split()
        if len(words) <= 120:
            sentences.append(cleaned)
            continue
        for start in range(0, len(words), 80):
            chunk = " ".join(words[start : start + 80])
            if chunk:
                sentences.append(chunk)
    return sentences


def build_context(prev_sent: str, current_sent: str, next_sent: str) -> tuple[str, int]:
    prefix = f"{prev_sent} " if prev_sent else ""
    middle = current_sent
    suffix = f" {next_sent}" if next_sent else ""
    context = f"{prefix}{middle}{suffix}".strip()
    return context, len(prefix)


def plot_distribution(distribution: pd.DataFrame, word: str) -> None:
    top = distribution.head(20).sort_values("occurrence_count")
    plt.figure(figsize=(11, 7))
    plt.barh(top["source_file"], top["occurrence_count"], color="#59A14F")
    plt.xlabel(f"Occurrences of '{word}'")
    plt.title(f"Documents with the most '{word}' occurrences")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "word_distribution.png", dpi=200)
    plt.close()



def main() -> None:
    ensure_output_dirs()
    word = load_selected_word()
    pattern = re.compile(rf"(?<![A-Za-z-]){re.escape(word)}(?![A-Za-z-])", re.IGNORECASE)

    occurrence_rows = []
    distribution_rows = []
    total_tokens = 0
    readable_documents = 0
    occurrence_id = 1

    for path in sorted(TEXTS_DIR.glob("*.txt")):
        raw_text = read_text(path)
        token_count = len(TOKEN_RE.findall(raw_text))
        total_tokens += token_count
        readable_documents += 1
        sentences = split_sentences(raw_text)

        doc_occurrence_count = 0
        for sent_index, sentence in enumerate(sentences):
            lower_sentence = sentence.lower()
            matches = list(pattern.finditer(lower_sentence))
            if not matches:
                continue

            prev_sent = sentences[sent_index - 1] if sent_index > 0 else ""
            next_sent = sentences[sent_index + 1] if sent_index + 1 < len(sentences) else ""
            context, prefix_length = build_context(prev_sent, sentence, next_sent)
            context_id = f"{path.stem}:{sent_index}"

            for match_index, match in enumerate(matches, start=1):
                doc_occurrence_count += 1
                occurrence_rows.append(
                    {
                        "occurrence_id": occurrence_id,
                        "context_id": context_id,
                        "selected_word": word,
                        "source_file": path.name,
                        "sentence_index": sent_index,
                        "match_index_in_sentence": match_index,
                        "target_text": sentence[match.start() : match.end()],
                        "target_start": prefix_length + match.start(),
                        "target_end": prefix_length + match.end(),
                        "previous_sentence": prev_sent,
                        "sentence_text": sentence,
                        "next_sentence": next_sent,
                        "context": context,
                        "context_word_count": len(context.split()),
                    }
                )
                occurrence_id += 1

        distribution_rows.append(
            {
                "source_file": path.name,
                "token_count": token_count,
                "occurrence_count": doc_occurrence_count,
            }
        )

    occurrences = pd.DataFrame(occurrence_rows)
    if occurrences.empty:
        raise ValueError(f"No occurrences found for selected word '{word}'.")

    distribution = pd.DataFrame(distribution_rows).sort_values(
        ["occurrence_count", "token_count"], ascending=[False, False]
    )
    corpus_overview = pd.DataFrame(
        [
            {
                "selected_word": word,
                "readable_documents": readable_documents,
                "total_tokens": total_tokens,
                "total_occurrences": int(len(occurrences)),
                "documents_with_word": int((distribution["occurrence_count"] > 0).sum()),
                "unit_of_analysis": "Sentence with target word plus one sentence before and after",
            }
        ]
    )

    examples = occurrences.sample(min(40, len(occurrences)), random_state=42).sort_values(
        ["source_file", "sentence_index", "match_index_in_sentence"]
    )

    corpus_overview.to_csv(OUTPUT_DIR / "corpus_overview.csv", index=False)
    occurrences.to_csv(OUTPUT_DIR / "occurrences.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "occurrence_examples.csv", index=False)
    distribution.to_csv(OUTPUT_DIR / "document_occurrence_distribution.csv", index=False)
    plot_distribution(distribution, word)


if __name__ == "__main__":
    main()
