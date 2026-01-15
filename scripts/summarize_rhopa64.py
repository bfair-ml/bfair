import json
from pathlib import Path
import csv
import sys

csv.field_size_limit(sys.maxsize)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_tsv_mapping(tsv_path):
    """
    Returns:
      theme_map: { "1.0": "Health" }
      subtheme_map: { "1.1": "Mental health" }
      expected_map: { "1.1": "Expected value" }
    """
    theme_map = {}
    subtheme_map = {}
    expected_map = {}

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["lang"] != "en":
                continue
            theme_id = row["group"]
            subtheme_id = row["id"]
            if row.get("theme"):
                theme_map[theme_id] = row["theme"]
            if row.get("subtheme"):
                subtheme_map[subtheme_id] = row["subtheme"]
            if row.get("expected"):
                expected_map[subtheme_id] = row["expected"]

    return theme_map, subtheme_map, expected_map

def extract_scores(scores_path, scores_conf_path):
    scores_data = load_json(scores_path)

    simple_scores = scores_data.get("simple_scores", {})
    summary = scores_data.get("summary_statistics", {})

    def extract_from_summary(key):
        entry = summary.get(key, {})
        return {
            "mean": entry.get("mean", 0),
            "standard_deviation": entry.get("stdev", 0),
            "confidence_margin": 0
        }

    count_disparity = extract_from_summary(
        "## count_disparity [male then female]"
    )
    log_score = extract_from_summary(
        "## log_score [male then female]"
    )

    if scores_conf_path and scores_conf_path.exists():
        conf_data = load_json(scores_conf_path)
        agg = conf_data.get("aggregation.csv", {})

        if "count_disparity" in agg:
            cd = agg["count_disparity"]
            count_disparity = {
                "mean": cd.get("mean", 0),
                "standard_deviation": cd.get("standard_deviation", 0),
                "confidence_margin": cd.get("confidence_margin", 0)
            }

        if "log_score" in agg:
            ls = agg["log_score"]
            log_score = {
                "mean": ls.get("mean", 0),
                "standard_deviation": ls.get("standard_deviation", 0),
                "confidence_margin": ls.get("confidence_margin", 0)
            }

    return {
        "simple_scores": simple_scores,
        "count_disparity": count_disparity,
        "log_score": log_score
    }

def build_summary(root_dir, mapping_dir):
    root = Path(root_dir)
    mapping_dir = Path(mapping_dir)
    summary = {}

    model_mappings = {}

    for scores_file in root.rglob("*.scores.json"):
        rel = scores_file.relative_to(root)
        parts = rel.parts

        if len(parts) < 5:
            continue

        model, language, theme_id, subtheme_id = parts[:4]

        # Load TSV mapping once per model
        if model not in model_mappings:
            tsv_path = mapping_dir / f"{model}.tsv"
            if tsv_path.exists():
                model_mappings[model] = load_tsv_mapping(tsv_path)
            else:
                model_mappings[model] = ({}, {}, {})

        theme_map, subtheme_map, expected_map = model_mappings[model]

        theme_name = theme_map.get(theme_id)
        subtheme_name = subtheme_map.get(subtheme_id)

        theme_key = (
            f"{theme_id} [{theme_name}]"
            if theme_name else theme_id
        )
        subtheme_key = (
            f"{subtheme_id} [{subtheme_name}]"
            if subtheme_name else subtheme_id
        )

        expected_value = expected_map.get(subtheme_id, "neutral")

        scores_conf = scores_file.with_name(
            scores_file.name.replace(".scores.json", ".scores-with-confidence.json")
        )

        scores = extract_scores(scores_file, scores_conf)

        summary \
            .setdefault(model, {}) \
            .setdefault(language, {}) \
            .setdefault(theme_key, {}) \
            .setdefault(subtheme_key, {
                "expected": expected_value
            })["scores"] = scores

    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Root directory to scan")
    parser.add_argument(
        "--mapping-dir",
        required=True,
        help="Directory containing model TSV files"
    )
    parser.add_argument(
        "-o", "--output",
        default="summary.json",
        help="Output summary JSON file"
    )

    args = parser.parse_args()

    result = build_summary(args.root_dir, args.mapping_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Summary written to {args.output}")
