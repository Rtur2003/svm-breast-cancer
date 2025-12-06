import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _compute_min_max(rows: Iterable[Sequence[float]]) -> Tuple[List[float], List[float]]:
    rows = list(rows)
    if not rows:
        raise ValueError("No feature rows provided for min-max calculation.")
    feature_count = len(rows[0])
    min_vals = [float("inf")] * feature_count
    max_vals = [float("-inf")] * feature_count

    for features in rows:
        for i in range(feature_count):
            min_vals[i] = min(min_vals[i], features[i])
            max_vals[i] = max(max_vals[i], features[i])
    return min_vals, max_vals


def preprocess_data(input_path: Path, output_path: Path) -> None:
    """
    Normalize numeric features and map diagnosis labels to {-1, 1}.
    """
    with input_path.open("r") as infile:
        reader = csv.reader(infile)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{input_path} is empty.")

        data = []
        for row in reader:
            if len(row) != len(header):
                continue
            id_val = row[0]
            diagnosis = 1 if row[1] == "M" else -1
            features = list(map(float, row[2:]))
            data.append((id_val, features, diagnosis))

    feature_vectors = [features for _, features, _ in data]
    min_vals, max_vals = _compute_min_max(feature_vectors)

    normalized = []
    for id_val, features, diagnosis in data:
        normalized_features = []
        for value, min_val, max_val in zip(features, min_vals, max_vals):
            if max_val - min_val == 0:
                normalized_features.append(0.0)
            else:
                normalized_features.append((value - min_val) / (max_val - min_val))
        normalized.append((id_val, normalized_features, diagnosis))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["id"] + header[2:] + ["label"])
        for id_val, features, label in normalized:
            writer.writerow([id_val] + features + [label])


def _default_paths() -> Tuple[Path, Path]:
    base = Path(__file__).resolve().parent
    return base / "data.csv", base / "processed_data.csv"


def main() -> None:
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Min-max normalize the breast cancer dataset.")
    parser.add_argument("--input", type=Path, default=default_input, help="Raw Kaggle CSV path.")
    parser.add_argument("--output", type=Path, default=default_output, help="Destination for normalized CSV.")
    args = parser.parse_args()

    preprocess_data(args.input, args.output)


if __name__ == "__main__":
    main()
