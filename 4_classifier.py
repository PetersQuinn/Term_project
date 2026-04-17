from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputsFinal"
FIGURES_DIR = OUTPUT_DIR / "figures"
RANDOM_SEED = 42

LABEL_SCHEMES = {
    "credit": {
        "belief_or_trust": {
            "keywords": {
                "beleeve",
                "believe",
                "belief",
                "credence",
                "faith",
                "knowledge",
                "opinion",
                "proof",
                "report",
                "reports",
                "scripture",
                "testimony",
                "truth",
                "true",
                "witness",
                "word",
            },
            "description": "credit as belief, trust, testimony, or acceptance of truth",
        },
        "reputation_or_status": {
            "keywords": {
                "approval",
                "authoritie",
                "authority",
                "discredit",
                "esteem",
                "estimation",
                "fame",
                "favour",
                "favor",
                "goodname",
                "honor",
                "honour",
                "name",
                "reputation",
                "worthy",
            },
            "description": "credit as reputation, standing, honor, or social authority",
        },
    },
    "trade": {
        "commercial_exchange": {
            "keywords": {
                "cargo",
                "commodity",
                "commodities",
                "commerce",
                "company",
                "consumption",
                "exchange",
                "foreign",
                "goods",
                "import",
                "imports",
                "india",
                "manufacture",
                "manufactures",
                "market",
                "merchant",
                "merchants",
                "merchandise",
                "money",
                "port",
                "ports",
                "price",
                "ship",
                "shipping",
                "traffic",
                "traffick",
                "traffique",
            },
            "description": "trade as commerce, exchange, shipping, or markets",
        },
        "occupation_or_craft": {
            "keywords": {
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
                "work",
                "workman",
                "workmanship",
            },
            "description": "trade as occupation, manual craft, or line of work",
        },
    }
}


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_selected_word() -> str:
    payload = json.loads((OUTPUT_DIR / "selected_word.json").read_text(encoding="utf-8"))
    return payload["selected_word"]


def assign_weak_label(text: str, scheme: dict) -> tuple[str | None, str]:
    lowered = text.lower()
    label_hits = {}
    for label_name, config in scheme.items():
        hits = sorted({keyword for keyword in config["keywords"] if re.search(rf"\b{re.escape(keyword)}\b", lowered)})
        label_hits[label_name] = hits

    ranked = sorted(label_hits.items(), key=lambda item: len(item[1]), reverse=True)
    best_label, best_hits = ranked[0]
    runner_up_hits = ranked[1][1]

    if not best_hits:
        return None, ""
    if len(best_hits) == len(runner_up_hits):
        return None, ""
    return best_label, ", ".join(best_hits[:6])


def mask_terms(text: str, selected_word: str, scheme: dict) -> str:
    lowered = text.lower()
    terms = {selected_word.lower()}
    for config in scheme.values():
        terms.update(config["keywords"])
    for term in sorted(terms, key=len, reverse=True):
        lowered = re.sub(rf"\b{re.escape(term)}\b", " ", lowered)
    return " ".join(lowered.split())


def plot_confusion(cm: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Classifier confusion matrix")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "classifier_confusion_matrix.png", dpi=220)
    plt.close()


def plot_top_features(model: LogisticRegression, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]
    top_positive_idx = coefficients.argsort()[-15:][::-1]
    top_negative_idx = coefficients.argsort()[:15]

    feature_frame = pd.DataFrame(
        {
            "feature": np.concatenate([feature_names[top_negative_idx], feature_names[top_positive_idx]]),
            "coefficient": np.concatenate([coefficients[top_negative_idx], coefficients[top_positive_idx]]),
        }
    ).sort_values("coefficient")

    plt.figure(figsize=(10, 7))
    colors = ["#E15759" if coef < 0 else "#4C78A8" for coef in feature_frame["coefficient"]]
    plt.barh(feature_frame["feature"], feature_frame["coefficient"], color=colors)
    plt.xlabel("Logistic regression coefficient")
    plt.title("Top classifier features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_features.png", dpi=220)
    plt.close()

    return feature_frame

def main() -> None:
    ensure_output_dirs()
    selected_word = load_selected_word()
    if selected_word not in LABEL_SCHEMES:
        raise ValueError(
            f"No classifier label scheme"
        )

    scheme = LABEL_SCHEMES[selected_word]
    occurrences = pd.read_csv(OUTPUT_DIR / "occurrences.csv")
    if (OUTPUT_DIR / "bert_cluster_assignments.csv").exists():
        bert_clusters = pd.read_csv(OUTPUT_DIR / "bert_cluster_assignments.csv")[
            ["occurrence_id", "cluster_id"]
        ]
        occurrences = occurrences.merge(bert_clusters, on="occurrence_id", how="left")

    labels = []
    reasons = []
    for text in occurrences["context"]:
        label, reason = assign_weak_label(text, scheme)
        labels.append(label)
        reasons.append(reason)

    occurrences["weak_label"] = labels
    occurrences["weak_label_reason"] = reasons
    labeled = occurrences.dropna(subset=["weak_label"]).copy()
    counts = labeled["weak_label"].value_counts().sort_index()

    if counts.min() < 25:
        raise ValueError("Weak labeling did not yield at least 25 examples per class.")

    sample_size = min(int(counts.min()), 250)
    labeled_subset = labeled.groupby("weak_label", group_keys=False).sample(
        n=sample_size, random_state=RANDOM_SEED
    )
    labeled_subset = labeled_subset.reset_index(drop=True)
    labeled_subset["model_text"] = labeled_subset["context"].map(lambda text: mask_terms(text, selected_word, scheme))

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        labeled_subset["model_text"],
        labeled_subset["weak_label"],
        labeled_subset.index,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=labeled_subset["weak_label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2500, random_state=RANDOM_SEED, class_weight="balanced")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)

    label_order = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=label_order)
    metrics = {
        "selected_word": selected_word,
        "labels": label_order,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": cm.tolist(),
        "weak_label_counts": counts.to_dict(),
        "balanced_training_examples_per_class": int(sample_size),
    }

    plot_confusion(cm, label_order)
    feature_frame = plot_top_features(model, vectorizer)
    feature_frame["direction"] = np.where(feature_frame["coefficient"] < 0, label_order[0], label_order[1])

    labeled_subset["split"] = "train"
    labeled_subset.loc[test_idx, "split"] = "test"
    labeled_subset.loc[test_idx, "predicted_label"] = y_pred
    labeled_subset.loc[test_idx, "prediction_probability"] = y_prob.max(axis=1)
    labeled_subset.loc[train_idx, "predicted_label"] = model.predict(vectorizer.transform(X_train))
    labeled_subset.loc[train_idx, "prediction_probability"] = model.predict_proba(vectorizer.transform(X_train)).max(axis=1)

    all_model_text = occurrences["context"].map(lambda text: mask_terms(text, selected_word, scheme))
    all_pred = model.predict(vectorizer.transform(all_model_text))
    all_prob = model.predict_proba(vectorizer.transform(all_model_text))
    prediction_frame = occurrences.copy()
    prediction_frame["predicted_label"] = all_pred
    for idx, label in enumerate(label_order):
        prediction_frame[f"prob_{label}"] = all_prob[:, idx]

    errors = labeled_subset.loc[test_idx].copy()
    errors = errors[errors["weak_label"] != errors["predicted_label"]].sort_values("prediction_probability")

    labeled_subset.to_csv(OUTPUT_DIR / "classifier_labeled_subset.csv", index=False)
    prediction_frame.to_csv(OUTPUT_DIR / "classifier_predictions.csv", index=False)
    errors.to_csv(OUTPUT_DIR / "classifier_error_examples.csv", index=False)
    feature_frame.to_csv(OUTPUT_DIR / "top_features.csv", index=False)
    (OUTPUT_DIR / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
