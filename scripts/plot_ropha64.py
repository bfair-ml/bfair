import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Data loading
# -----------------------------


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_summary(summary):
    rows = []

    for model, langs in summary.items():
        for lang, themes in langs.items():
            for theme, subthemes in themes.items():
                for subtheme, data in subthemes.items():
                    scores = data["scores"]
                    rows.append(
                        {
                            "model": model,
                            "language": lang,
                            "theme": theme,
                            "subtheme": subtheme,
                            "expected": data["expected"],
                            "simple_male": scores["simple_scores"].get("male"),
                            "simple_female": scores["simple_scores"].get("female"),
                            "log_mean": scores["log_score"]["mean"],
                            "log_conf": scores["log_score"]["confidence_margin"],
                            "count_disparity": scores["count_disparity"]["mean"],
                        }
                    )

    return pd.DataFrame(rows)


def assign_subtheme_roles(df):
    """
    Assigns roles:
      female-1, female-2, male-1, male-2
    per (model, language, theme)

    Subthemes with expected == 'neutral' (e.g. 'all') get NaN.
    """
    df = df.copy()
    df["role"] = None

    for (model, language, theme), group in df.groupby(["model", "language", "theme"]):
        # exclude neutral expectations (e.g. "all")
        male = group[group.expected == "male"].sort_values("subtheme")
        female = group[group.expected == "female"].sort_values("subtheme")

        for i, idx in enumerate(female.index[:2]):
            df.at[idx, "role"] = f"female-{i+1}"
        for i, idx in enumerate(male.index[:2]):
            df.at[idx, "role"] = f"male-{i+1}"

    return df


# -----------------------------
# Bias logic
# -----------------------------


def observed_bias(row, metric, threshold):
    if metric == "log_score":
        if abs(row.log_mean) < threshold:
            return "neutral"
        return "male" if row.log_mean > 0 else "female"

    if metric == "simple_scores":
        m, f = row.simple_male, row.simple_female
        if pd.isna(m) or pd.isna(f):
            return "male" if pd.notna(m) else "female"
        ratio = max(m, f) / min(m, f)
        if ratio < threshold:
            return "neutral"
        return "male" if m > f else "female"

    return None


def classify_alignment(expected, observed):
    if observed == "neutral":
        return "neutral"
    if expected == "neutral":
        return "biased"
    if expected == observed:
        return "stereotypical"
    return "counter"


# -----------------------------
# Utilities
# -----------------------------

ALIGNMENT_MAP = {"stereotypical": 1, "biased": 0.5, "neutral": 0, "counter": -1}


def save_or_show(fig, path=None):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def filename_from_args(args, model, language):
    parts = [args.plot, args.metric, model, language or "all_languages"]
    if args.expected:
        parts.append("expected_" + "-".join(args.expected))
    return "_".join(parts) + ".png"


# -----------------------------
# Plotting
# -----------------------------


ROLE_ORDER = ["female-1", "female-2", "male-1", "male-2"]


def plot_heatmap(df, args, model, language):
    sub = df[df.model == model]

    if language:
        sub = sub[sub.language == language]
    if args.expected:
        sub = sub[sub.expected.isin(args.expected)]

    sub["observed"] = sub.apply(
        observed_bias, axis=1, metric=args.metric, threshold=args.threshold
    )

    sub["category"] = sub.apply(
        lambda r: classify_alignment(r.expected, r.observed), axis=1
    )

    if args.heatmap_mode == "roles":
        # Exclude neutral expectations (e.g. "all")
        sub = sub[sub.role.notna()]

        pivot = (
            sub.pivot(index="theme", columns="role", values="category")
            .reindex(columns=ROLE_ORDER)
            .replace(ALIGNMENT_MAP)
        )
    else:
        pivot = sub.pivot(index="theme", columns="subtheme", values="category").replace(
            ALIGNMENT_MAP
        )

    fig = plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, cmap="RdBu", center=0, annot=True, fmt=".1f", linewidths=0.5)

    mode_label = "roles" if args.heatmap_mode == "roles" else "subthemes"
    plt.title(
        f"{model} | {language or 'all languages'} | "
        f"{args.metric} | heatmap={mode_label}"
    )
    plt.tight_layout()

    return fig


def plot_alignment_bars(df, args, model, language):
    sub = df[(df.model == model)]
    if language:
        sub = sub[sub.language == language]

    sub["observed"] = sub.apply(
        observed_bias, axis=1, metric=args.metric, threshold=args.threshold
    )
    sub["category"] = sub.apply(
        lambda r: classify_alignment(r.expected, r.observed), axis=1
    )

    counts = sub.groupby(["theme", "category"]).size().reset_index(name="count")

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=counts, x="theme", y="count", hue="category")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{model} | Bias alignment by theme")
    plt.tight_layout()

    return fig


def plot_count_disparity(df, args, model, language):
    sub = df[(df.model == model)]
    if language:
        sub = sub[sub.language == language]

    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(data=sub, x="theme", y="count_disparity")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{model} | Count disparity")
    plt.tight_layout()

    return fig


# -----------------------------
# CLI
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Bias analysis and visualization CLI")

    parser.add_argument("summary_json")
    parser.add_argument(
        "--plot", choices=["heatmap", "bars", "count-disparity"], required=True
    )
    parser.add_argument(
        "--heatmap-mode",
        choices=["subthemes", "roles"],
        default="subthemes",
        help="Heatmap columns: raw subthemes or semantic roles",
    )
    parser.add_argument(
        "--metric",
        choices=["log_score", "simple_scores", "count_disparity"],
        default="log_score",
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--model")
    parser.add_argument("--language")
    parser.add_argument("--expected", nargs="+", choices=["male", "female", "neutral"])
    parser.add_argument("--save", type=Path, help="Directory to save figures")

    args = parser.parse_args()

    df = flatten_summary(load_summary(args.summary_json))

    df = assign_subtheme_roles(df)

    models = [args.model] if args.model else sorted(df.model.unique())
    languages = [args.language] if args.language else sorted(df.language.unique())

    for model in models:
        for language in languages:
            if args.plot == "heatmap":
                fig = plot_heatmap(df, args, model, language)
            elif args.plot == "bars":
                fig = plot_alignment_bars(df, args, model, language)
            else:
                fig = plot_count_disparity(df, args, model, language)

            path = None
            if args.save:
                fname = filename_from_args(args, model, language)
                path = args.save / fname

            save_or_show(fig, path)


if __name__ == "__main__":
    main()
