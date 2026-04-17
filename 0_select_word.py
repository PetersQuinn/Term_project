from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
TEXTS_DIR = PROJECT_ROOT / "texts"
OUTPUT_DIR = PROJECT_ROOT / "outputsFinal"
FIGURES_DIR = OUTPUT_DIR / "figures"
RANDOM_SEED = 42

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")

STOP_WORDS = set(ENGLISH_STOP_WORDS)
STOP_WORDS |= {
    "also",
    "article",
    "articles",
    "beene",
    "book",
    "came",
    "could",
    "day",
    "days",
    "doth",
    "euen",
    "every",
    "first",
    "fourth",
    "gaue",
    "giue",
    "god",
    "good",
    "great",
    "haue",
    "hath",
    "hereof",
    "himselfe",
    "item",
    "king",
    "like",
    "little",
    "lord",
    "majesty",
    "man",
    "many",
    "much",
    "must",
    "ninth",
    "onely",
    "owne",
    "person",
    "persons",
    "said",
    "saith",
    "say",
    "second",
    "seventh",
    "shall",
    "sixth",
    "tenth",
    "thee",
    "their",
    "themselues",
    "thereby",
    "therein",
    "thereof",
    "thereunto",
    "third",
    "thing",
    "things",
    "thou",
    "thy",
    "time",
    "times",
    "unto",
    "upon",
    "vnto",
    "vpon",
    "whether",
    "wherein",
    "whereof",
    "within",
    "without",
    "woman",
    "women",
    "would",
    "year",
    "years",
    "yet",
}

CONCEPT_HINTS = {
    "authority",
    "capital",
    "commerce",
    "credit",
    "debt",
    "freedom",
    "industry",
    "interest",
    "justice",
    "labor",
    "law",
    "liberty",
    "market",
    "order",
    "peace",
    "power",
    "property",
    "religion",
    "safety",
    "security",
    "state",
    "trade",
    "value",
    "virtue",
    "wealth",
}

CONCEPTUAL_SUFFIXES = (
    "acy",
    "age",
    "ance",
    "ence",
    "dom",
    "hood",
    "ion",
    "ism",
    "ity",
    "ment",
    "ness",
    "ship",
    "ture",
)

DOMAIN_LEXICONS = {
    "commerce": {
        "cargo",
        "commodity",
        "commodities",
        "commerce",
        "company",
        "exchange",
        "export",
        "foreign",
        "goods",
        "imports",
        "india",
        "manufacture",
        "manufactures",
        "market",
        "merchant",
        "merchants",
        "merchandise",
        "money",
        "price",
        "ship",
        "shipping",
        "trade",
        "traffic",
        "traffick",
        "traffique",
    },
    "occupation": {
        "apprentice",
        "art",
        "calling",
        "craft",
        "education",
        "follow",
        "labour",
        "labor",
        "manual",
        "occupation",
        "parents",
        "profession",
        "skill",
        "trade",
        "work",
        "workman",
        "workmanship",
    },
    "governance": {
        "authority",
        "commonwealth",
        "council",
        "court",
        "empire",
        "government",
        "king",
        "kingdom",
        "law",
        "parliament",
        "peace",
        "policy",
        "power",
        "prince",
        "state",
    },
    "religion": {
        "christ",
        "church",
        "conscience",
        "faith",
        "gospel",
        "god",
        "grace",
        "holy",
        "mercy",
        "pope",
        "religion",
        "sacrament",
        "sin",
        "spirit",
        "truth",
    },
    "geography": {
        "city",
        "coast",
        "country",
        "east",
        "england",
        "europe",
        "harbour",
        "harbor",
        "kingdom",
        "london",
        "nation",
        "ocean",
        "place",
        "port",
        "sea",
        "town",
        "west",
    },
}


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_token(token: str) -> str:
    return token.lower().strip("'").replace("'", "")


def token_ok(token: str) -> bool:
    if len(token) < 4 or len(token) > 16:
        return False
    if token in STOP_WORDS:
        return False
    if not token.isalpha():
        return False
    if token.endswith("eth") or token.endswith("est"):
        return False
    if len(set(token)) == 1:
        return False
    return True


def concept_like(token: str) -> bool:
    return token in CONCEPT_HINTS or token.endswith(CONCEPTUAL_SUFFIXES)


def split_sentences(text: str) -> list[str]:
    pieces = []
    for part in SENTENCE_SPLIT_RE.split(text.replace("\r", "\n")):
        cleaned = " ".join(part.split())
        if not cleaned:
            continue
        words = cleaned.split()
        if len(words) <= 120:
            pieces.append(cleaned)
            continue
        for start in range(0, len(words), 80):
            chunk = " ".join(words[start : start + 80])
            if chunk:
                pieces.append(chunk)
    return pieces


def first_pass(paths: list[Path]) -> pd.DataFrame:
    total_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()

    for path in paths:
        tokens = [normalize_token(token) for token in TOKEN_RE.findall(read_text(path))]
        usable = [token for token in tokens if token_ok(token)]
        total_counts.update(usable)
        doc_counts.update(set(usable))

    rows = []
    for word, total in total_counts.items():
        doc_count = doc_counts[word]
        if total < 350 or doc_count < 40:
            continue
        if not concept_like(word):
            continue
        count_score = max(0.0, 1 - abs(math.log(total) - math.log(4000)) / 2.2)
        spread_score = min(doc_count / 350, 1.0)
        hint_bonus = 0.14 if word in CONCEPT_HINTS else 0.0
        penalty = 0.15 if total > 16000 else 0.0
        base_score = 0.55 * count_score + 0.45 * spread_score + hint_bonus - penalty
        rows.append(
            {
                "word": word,
                "occurrences": int(total),
                "document_count": int(doc_count),
                "base_score": round(base_score, 6),
                "concept_hint": int(word in CONCEPT_HINTS),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["base_score", "occurrences"], ascending=[False, False])
    return frame


def collect_details(paths: list[Path], shortlisted_words: list[str]) -> tuple[dict[str, dict], list[dict]]:
    patterns = {
        word: re.compile(rf"(?<![A-Za-z-]){re.escape(word)}(?![A-Za-z-])", re.IGNORECASE)
        for word in shortlisted_words
    }
    details = {
        word: {
            "neighbor_counts": Counter(),
            "capital_total": 0,
            "example_contexts": [],
            "domain_hits": Counter(),
        }
        for word in shortlisted_words
    }

    for path in paths:
        raw_text = read_text(path)
        raw_tokens = TOKEN_RE.findall(raw_text)
        tokens = [normalize_token(token) for token in raw_tokens]

        for index, token in enumerate(tokens):
            if token not in details:
                continue
            if index > 0:
                details[token]["neighbor_counts"][f"L:{tokens[index - 1]}"] += 1
            if index + 1 < len(tokens):
                details[token]["neighbor_counts"][f"R:{tokens[index + 1]}"] += 1
            window = raw_tokens[max(0, index - 8) : index + 9]
            capital_hits = sum(1 for piece in window if len(piece) > 1 and piece[0].isupper())
            details[token]["capital_total"] += capital_hits

        sentences = split_sentences(raw_text)
        for sent_index, sentence in enumerate(sentences):
            lower_sentence = sentence.lower()
            for word, pattern in patterns.items():
                if len(details[word]["example_contexts"]) >= 12 and sent_index > 0:
                    continue
                if not pattern.search(lower_sentence):
                    continue
                prev_sent = sentences[sent_index - 1] if sent_index > 0 else ""
                next_sent = sentences[sent_index + 1] if sent_index + 1 < len(sentences) else ""
                context = " ".join(part for part in [prev_sent, sentence, next_sent] if part).strip()
                context = " ".join(context.split())
                if len(details[word]["example_contexts"]) < 12:
                    details[word]["example_contexts"].append(
                        {"word": word, "source_file": path.name, "context": context}
                    )
                context_tokens = {normalize_token(token) for token in TOKEN_RE.findall(context)}
                for domain_name, lexicon in DOMAIN_LEXICONS.items():
                    if context_tokens & lexicon:
                        details[word]["domain_hits"][domain_name] += 1

    example_rows = []
    for word in shortlisted_words:
        for rank, example in enumerate(details[word]["example_contexts"], start=1):
            example_rows.append(
                {
                    "word": word,
                    "example_rank": rank,
                    "source_file": example["source_file"],
                    "context": example["context"],
                }
            )

    return details, example_rows


def score_candidates(base_frame: pd.DataFrame, details: dict[str, dict]) -> pd.DataFrame:
    rows = []

    for row in base_frame.to_dict("records"):
        word = row["word"]
        info = details[word]
        neighbor_counts = info["neighbor_counts"]
        total_neighbors = max(sum(neighbor_counts.values()), 1)
        entropy = -sum(
            (count / total_neighbors) * math.log(count / total_neighbors, 2)
            for count in neighbor_counts.values()
        )
        unique_neighbors = len(neighbor_counts)
        avg_capital_hits = info["capital_total"] / max(row["occurrences"], 1)
        domain_hits = info["domain_hits"]
        active_domains = sum(1 for value in domain_hits.values() if value > 0)
        top_domains = ", ".join(
            f"{domain}:{count}" for domain, count in domain_hits.most_common(3) if count > 0
        )
        diversity_score = 0.6 * min(entropy / 7.4, 1.0) + 0.4 * min(
            math.log10(unique_neighbors + 1) / math.log10(1800 + 1), 1.0
        )
        manageability = 1.0 if row["occurrences"] <= 9000 else max(0.2, 1 - (row["occurrences"] - 9000) / 15000)
        ner_score = min(avg_capital_hits / 2.0, 1.0)
        domain_score = min(active_domains / 3.0, 1.0)
        final_score = (
            0.42 * row["base_score"]
            + 0.20 * diversity_score
            + 0.12 * manageability
            + 0.14 * ner_score
            + 0.12 * domain_score
        )
        rows.append(
            {
                **row,
                "neighbor_entropy": round(entropy, 4),
                "unique_neighbors": int(unique_neighbors),
                "avg_capital_hits": round(avg_capital_hits, 4),
                "active_domains": int(active_domains),
                "top_domains": top_domains,
                "final_score": round(final_score, 6),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["final_score", "occurrences"], ascending=[False, False])
    return frame


def choose_recommended_word(frame: pd.DataFrame) -> str:
    practical = frame[
        (frame["occurrences"].between(1500, 9000))
        & (frame["document_count"] >= 120)
        & (frame["active_domains"] >= 2)
        & (frame["avg_capital_hits"] >= 1.4)
        & (frame["concept_hint"] == 1)
    ].copy()
    if practical.empty:
        return frame.iloc[0]["word"]

    practical["practical_score"] = (
        practical["final_score"]
        + 0.05 * practical["active_domains"]
        + 0.03 * practical["avg_capital_hits"]
    )
    preferred_order = ["trade", "credit", "authority", "wealth", "liberty", "justice"]
    practical = practical.sort_values(["practical_score", "occurrences"], ascending=[False, False])

    for preferred in preferred_order:
        match = practical[practical["word"] == preferred]
        if not match.empty:
            return match.iloc[0]["word"]
    return practical.iloc[0]["word"]


def plot_candidate_scores(frame: pd.DataFrame) -> None:
    top = frame.head(12).sort_values("final_score")
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top["word"], top["final_score"], color="#4C78A8")
    for bar, value in zip(bars, top["final_score"]):
        plt.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    plt.xlabel("Heuristic score")
    plt.title("Top concept candidates")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "concept_candidate_scores.png", dpi=200)
    plt.close()


def main() -> None:
    random.seed(RANDOM_SEED)
    ensure_output_dirs()
    paths = sorted(TEXTS_DIR.glob("*.txt"))
    if not paths:
        raise FileNotFoundError("No .txt files found in texts/.")

    base_frame = first_pass(paths)
    top_base = base_frame.head(40).copy()
    shortlisted_words = top_base["word"].tolist()
    details, example_rows = collect_details(paths, shortlisted_words)
    scored_frame = score_candidates(top_base, details)
    recommended_word = choose_recommended_word(scored_frame)

    examples_frame = pd.DataFrame(example_rows).sort_values(["word", "example_rank"])
    scored_frame.to_csv(OUTPUT_DIR / "candidate_words.csv", index=False)
    examples_frame.to_csv(OUTPUT_DIR / "candidate_word_examples.csv", index=False)
    plot_candidate_scores(scored_frame)

if __name__ == "__main__":
    main()
