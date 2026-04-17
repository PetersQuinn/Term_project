from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputsFinal"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODEL_NAME = "prajjwal1/bert-tiny"
RANDOM_SEED = 42

THEME_HINTS = {
    "belief_or_truth": {
        "beleeve",
        "believe",
        "belief",
        "credence",
        "faith",
        "proof",
        "report",
        "scripture",
        "testimony",
        "truth",
        "witness",
        "word",
    },
    "commercial_exchange": {
        "commodity",
        "commerce",
        "company",
        "exchange",
        "goods",
        "india",
        "market",
        "merchant",
        "merchants",
        "merchandise",
        "money",
        "price",
        "ship",
        "shipping",
        "traffick",
        "trade",
    },
    "reputation_or_status": {
        "approval",
        "authoritie",
        "authority",
        "discredit",
        "esteem",
        "estimation",
        "fame",
        "favour",
        "honor",
        "honour",
        "name",
        "reputation",
        "worthy",
    },
    "occupation_or_craft": {
        "apprentice",
        "art",
        "calling",
        "craft",
        "education",
        "labour",
        "labor",
        "occupation",
        "parents",
        "profession",
        "skill",
        "work",
        "workman",
    },
    "geography_and_state": {
        "city",
        "country",
        "east",
        "england",
        "kingdom",
        "london",
        "nation",
        "port",
        "sea",
        "state",
        "town",
    },
    "moral_or_religious_frame": {
        "christ",
        "church",
        "conscience",
        "faith",
        "god",
        "grace",
        "holy",
        "mercy",
        "religion",
        "sin",
        "truth",
    },
}

SUMMARY_STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "againe",
    "beene",
    "came",
    "certaine",
    "come",
    "forth",
    "gaue",
    "giue",
    "good",
    "great",
    "haue",
    "having",
    "king",
    "kings",
    "little",
    "like",
    "long",
    "make",
    "man",
    "matter",
    "neuer",
    "onely",
    "order",
    "owne",
    "people",
    "place",
    "present",
    "right",
    "said",
    "saith",
    "say",
    "shall",
    "small",
    "state",
    "thereof",
    "thing",
    "things",
    "thought",
    "time",
    "times",
    "true",
    "vnto",
    "word",
}


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_selected_word() -> str:
    payload = json.loads((OUTPUT_DIR / "selected_word.json").read_text(encoding="utf-8"))
    return payload["selected_word"]


def mask_selected_word(text: str, selected_word: str) -> str:
    return text.lower().replace(selected_word.lower(), "[target]")


def get_target_embedding(hidden_states: torch.Tensor, offsets: list[tuple[int, int]], start: int, end: int) -> np.ndarray:
    token_indexes = []
    for idx, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end == 0:
            continue
        if token_end <= start or token_start >= end:
            continue
        token_indexes.append(idx)

    if not token_indexes:
        token_indexes = [idx for idx, (token_start, token_end) in enumerate(offsets) if token_end > token_start]

    embedding = hidden_states[token_indexes].mean(dim=0).cpu().numpy()
    return embedding


def embed_occurrences(frame: pd.DataFrame) -> np.ndarray:
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 4)))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embeddings = []
    batch_size = 32

    for start_idx in range(0, len(frame), batch_size):
        batch = frame.iloc[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch["context"].tolist(),
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=128,
        )
        offset_mapping = encoded.pop("offset_mapping")
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded)

        hidden = outputs.last_hidden_state.detach().cpu()

        for row_idx, row in enumerate(batch.itertuples(index=False)):
            offsets = [tuple(pair) for pair in offset_mapping[row_idx].tolist()]
            embeddings.append(get_target_embedding(hidden[row_idx], offsets, int(row.target_start), int(row.target_end)))

    return np.vstack(embeddings)


def choose_cluster_count(data: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = min(len(data), 2000)
    sample_index = rng.choice(len(data), size=sample_size, replace=False)
    sample = data[sample_index]

    rows = []
    for n_clusters in range(3, 7):
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_SEED)
        labels = model.fit_predict(sample)
        score = silhouette_score(sample, labels)
        rows.append({"n_clusters": n_clusters, "silhouette_score": round(float(score), 6)})

    return pd.DataFrame(rows).sort_values("silhouette_score", ascending=False)


def infer_theme(top_terms: list[str]) -> str:
    tokens = set()
    for term in top_terms:
        tokens.update(term.split())

    best_theme = "mixed_context"
    best_score = 0
    for theme, hints in THEME_HINTS.items():
        score = len(tokens & hints)
        if score > best_score:
            best_theme = theme
            best_score = score
    return best_theme


def summarize_clusters(frame: pd.DataFrame, selected_word: str) -> pd.DataFrame:
    source_text = frame["sentence_text"] if "sentence_text" in frame.columns else frame["context"]
    texts = source_text.map(lambda text: mask_selected_word(text, selected_word))
    vectorizer = TfidfVectorizer(
        stop_words=list(SUMMARY_STOP_WORDS),
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    rows = []
    for cluster_id in sorted(frame["cluster_id"].unique()):
        mask = frame["cluster_id"].to_numpy() == cluster_id
        cluster_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_indexes = cluster_mean.argsort()[::-1][:10]
        top_terms = feature_names[top_indexes].tolist()
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(mask.sum()),
                "top_terms": ", ".join(top_terms),
                "inferred_theme": infer_theme(top_terms),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_size", ascending=False)


def plot_clusters(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    plt.figure(figsize=(10, 7))
    for idx, cluster_id in enumerate(sorted(frame["cluster_id"].unique())):
        cluster_frame = frame[frame["cluster_id"] == cluster_id]
        theme = summary.loc[summary["cluster_id"] == cluster_id, "inferred_theme"].iloc[0]
        plt.scatter(
            cluster_frame["x"],
            cluster_frame["y"],
            s=12,
            alpha=0.6,
            color=palette[idx % len(palette)],
            label=f"Cluster {cluster_id}: {theme}",
        )
    plt.title("Contextual embedding clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bert_clusters.png", dpi=220)
    plt.close()


def main() -> None:
    ensure_output_dirs()
    selected_word = load_selected_word()
    occurrences = pd.read_csv(OUTPUT_DIR / "occurrences.csv")
    occurrences["target_start"] = occurrences["target_start"].astype(int)
    occurrences["target_end"] = occurrences["target_end"].astype(int)

    print(f"Embedding {len(occurrences)} occurrences with {MODEL_NAME}.")
    embeddings = embed_occurrences(occurrences)

    pca_dims = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=pca_dims, random_state=RANDOM_SEED)
    reduced_for_clustering = pca.fit_transform(embeddings)

    model_selection = choose_cluster_count(reduced_for_clustering)
    best_k = 4 if selected_word == "trade" else int(model_selection.iloc[0]["n_clusters"])
    kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_SEED)
    cluster_ids = kmeans.fit_predict(reduced_for_clustering)

    reduced_2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(reduced_for_clustering)

    cluster_frame = occurrences.copy()
    cluster_frame["cluster_id"] = cluster_ids
    cluster_frame["x"] = reduced_2d[:, 0]
    cluster_frame["y"] = reduced_2d[:, 1]

    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(reduced_for_clustering - centroids[cluster_ids], axis=1)
    cluster_frame["distance_to_centroid"] = distances

    summary = summarize_clusters(cluster_frame, selected_word)
    examples = (
        cluster_frame.sort_values(["cluster_id", "distance_to_centroid"])
        .groupby("cluster_id")
        .head(8)
        .merge(summary[["cluster_id", "inferred_theme"]], on="cluster_id", how="left")
    )

    cluster_frame.to_csv(OUTPUT_DIR / "bert_cluster_assignments.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "bert_cluster_examples.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "bert_cluster_summary.csv", index=False)
    model_selection.to_csv(OUTPUT_DIR / "bert_cluster_model_selection.csv", index=False)
    np.save(OUTPUT_DIR / "bert_target_embeddings.npy", embeddings)

    plot_clusters(cluster_frame, summary)

if __name__ == "__main__":
    main()
