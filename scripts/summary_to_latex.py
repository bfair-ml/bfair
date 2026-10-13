import json
import argparse
import pandas as pd
from plot_ropha64 import classify_alignment, observed_bias


DEFAULT_METRIC = "log_score"
DEFAULT_THRESHOLD = 0.2

NON_NEUTRAL_ALIGNMENT_LABELS = [
    ("stereotypical", "S"),
    ("counter", "C"),
    ("neutral", "N"),
]

NEUTRAL_ALIGNMENT_LABELS = [
    ("female-biased", "F"),
    ("male-biased", "M"),
    ("neutral", "N"),
]


def compute_expectation_summaries(summary, metric, threshold):
    grouped_rows = []

    for model, langs in summary.items():
        for language, themes in langs.items():
            non_neutral_counts = {"stereotypical": 0, "counter": 0, "neutral": 0}
            neutral_counts = {"female-biased": 0, "male-biased": 0, "neutral": 0}

            for theme, subthemes in themes.items():
                if theme == "all":
                    continue

                for subtheme, details in subthemes.items():
                    if not isinstance(details, dict) or "expected" not in details:
                        continue

                    scores = details.get("scores", {})
                    score_row = pd.Series(
                        {
                            "log_mean": scores.get("log_score", {}).get("mean"),
                            "simple_male": scores.get("simple_scores", {}).get("male"),
                            "simple_female": scores.get("simple_scores", {}).get(
                                "female"
                            ),
                        }
                    )

                    observed = observed_bias(
                        score_row, metric=metric, threshold=threshold
                    )
                    category = classify_alignment(details["expected"], observed)

                    if details["expected"] in {
                        "male",
                        "female",
                    }:  # e.i., subtheme != "all"
                        non_neutral_counts[category] += 1
                    elif details["expected"] == "neutral":  # e.i., subtheme == "all"
                        neutral_counts[category] += 1

            grouped_rows.append(
                {
                    "Model": model,
                    "Language": language.capitalize(),
                    "Non-neutral Expectations": ", ".join(
                        f"{label}: {non_neutral_counts[name]}"
                        for name, label in NON_NEUTRAL_ALIGNMENT_LABELS
                    ),
                    "Neutral Expectations": ", ".join(
                        f"{label}: {neutral_counts[name]}"
                        for name, label in NEUTRAL_ALIGNMENT_LABELS
                    ),
                }
            )

    return pd.DataFrame(grouped_rows)


def convert_json_to_latex(
    input_json_path,
    output_tex_path,
    metric=DEFAULT_METRIC,
    threshold=DEFAULT_THRESHOLD,
    just_global=False,
):
    # Load the JSON data
    with open(input_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []

    # Traverse the nested structure: Model -> Language -> Category -> Subcategory
    for model, langs in data.items():
        for lang, categories in langs.items():
            # Sometimes the structure has a global 'all' key at this level
            if lang == "all":
                continue

            for cat, subcats in categories.items():
                if cat != "all" and just_global:
                    continue

                for subcat, details in subcats.items():
                    if not isinstance(details, dict) or "expected" not in details:
                        continue

                    expected = details.get("expected", "N/A")
                    scores = details.get("scores", {})

                    # Extract simple scores and format them as a readable string
                    simple_scores = scores.get("simple_scores", {})
                    simple_scores_str = ", ".join(
                        [f"{k}: {v:.2f}" for k, v in simple_scores.items()]
                    )

                    # Extract mean values for other metrics
                    count_disp_mean = scores.get("count_disparity", {}).get(
                        "mean", None
                    )
                    log_score_mean = scores.get("log_score", {}).get("mean", None)

                    # Append flattened data
                    rows.append(
                        {
                            "Model": model,
                            "Language": lang.capitalize(),
                            "Category": cat,
                            "Subcategory": subcat,
                            "Expected": expected,
                            "Simple Scores": simple_scores_str,
                            "Count Disp. (Mean)": (
                                round(count_disp_mean, 4)
                                if count_disp_mean is not None
                                else "N/A"
                            ),
                            "Log Score (Mean)": (
                                round(log_score_mean, 4)
                                if log_score_mean is not None
                                else "N/A"
                            ),
                        }
                    )

    # Create a pandas DataFrame and compute grouped alignment summaries from the raw JSON.
    df = pd.DataFrame(rows)
    summary_df = compute_expectation_summaries(data, metric, threshold)
    df = df.merge(summary_df, on=["Model", "Language"], how="left")

    # Export the DataFrame to a LaTeX table
    # Note: For pandas 2.0+, styling is preferred, but to_latex works for basic structures.
    latex_table = df.to_latex(
        index=False,
        longtable=True,
        caption="Summary of Model Scores by Category and Language",
        label="tab:summary_scores",
        column_format="p{2.5cm} p{2.2cm} p{3.2cm} p{4.0cm} c p{4.2cm} c c p{3.6cm} p{3.6cm}",
    )

    # Write to a .tex file
    with open(output_tex_path, "w", encoding="utf-8") as file:
        file.write(latex_table)

    print(f"Successfully processed {len(df)} rows.")
    print(f"LaTeX table saved to {output_tex_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert a fairness summary JSON file into a LaTeX table."
    )
    parser.add_argument(
        "input_json_path",
        nargs="?",
        default="summary.json",
        help="Path to the input summary JSON file.",
    )
    parser.add_argument(
        "output_tex_path",
        nargs="?",
        default="summary_table.tex",
        help="Path to the output LaTeX file.",
    )
    parser.add_argument(
        "--metric",
        choices=["log_score", "simple_scores"],
        default=DEFAULT_METRIC,
        help="Metric used to classify observed bias when computing the summary counts.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold used by the selected metric to determine neutrality.",
    )
    parser.add_argument(
        "--just-global",
        action="store_true",
        help="If set, only processes the 'all' category for each language, ignoring specific themes.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    convert_json_to_latex(
        args.input_json_path,
        args.output_tex_path,
        metric=args.metric,
        threshold=args.threshold,
        just_global=args.just_global,
    )


if __name__ == "__main__":
    main()
