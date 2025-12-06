import csv
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

Example = Tuple[str, List[float], int]


def load_processed_data(filepath: Union[str, Path]) -> List[Example]:
    """
    Read the normalized dataset and return (id, features, label) tuples.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("r", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)  # header
        data: List[Example] = []
        for row in reader:
            id_val = row[0]
            *features, label = row[1:]
            data.append((id_val, list(map(float, features)), int(label)))
    if not data:
        raise ValueError(f"No rows found in {path}")
    return data


def split_data(
    data: Sequence[Example],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """
    Deterministically split the dataset into train/val/test partitions.
    """
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    return train_data, val_data, test_data


def export_predictions(output_path: Union[str, Path], rows: Iterable[Tuple[str, int, int]]) -> None:
    """
    Write prediction rows to CSV with the expected header.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "true_label", "predicted_label"])
        writer.writerows(rows)
