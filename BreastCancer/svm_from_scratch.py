import argparse
import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from data_utils import export_predictions, load_processed_data, split_data


Example = Tuple[str, List[float], int]


def dot_product(x: Sequence[float], w: Sequence[float]) -> float:
    return sum(xi * wi for xi, wi in zip(x, w))


def predict(x: Sequence[float], w: Sequence[float], b: float) -> int:
    return 1 if dot_product(x, w) + b >= 0 else -1


def compute_loss(data: Sequence[Example], w: Sequence[float], b: float, C: float) -> float:
    loss = 0.0
    for _, x, y in data:
        margin = y * (dot_product(x, w) + b)
        loss += max(0.0, 1 - margin)
    regularization = 0.5 * sum(wi**2 for wi in w)
    return regularization + C * loss


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == -1 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == -1)

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def train_svm(
    train_data: Sequence[Example],
    epochs: int = 100,
    lr: float = 0.001,
    C: float = 10.0,
    seed: int = 42,
) -> Tuple[List[float], float]:
    feature_len = len(train_data[0][1])
    w = [0.0] * feature_len
    b = 0.0
    rng = random.Random(seed)
    samples = list(train_data)

    for epoch in range(1, epochs + 1):
        rng.shuffle(samples)

        for _, x, y in samples:
            margin = y * (dot_product(x, w) + b)
            if margin >= 1:
                w = [wi - lr * wi for wi in w]
            else:
                w = [wi - lr * (wi - C * y * xi) for wi, xi in zip(w, x)]
                b += lr * C * y

        loss = compute_loss(samples, w, b, C)
        predictions = [predict(x, w, b) for _, x, _ in samples]
        labels = [y for _, _, y in samples]
        f1 = f1_score(labels, predictions)
        print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Train F1 = {f1:.4f}")

    norm_w = math.sqrt(sum(wi**2 for wi in w))
    print(f"Final ||w|| = {norm_w:.4f}")
    return w, b


def evaluate(data: Sequence[Example], w: Sequence[float], b: float, output_path: Optional[Path] = None):
    predictions = []
    labels = []
    results = []

    for id_val, x, y in data:
        pred = predict(x, w, b)
        predictions.append(pred)
        labels.append(y)
        results.append((id_val, y, pred))

    correct = sum(1 for y_true, y_pred in zip(labels, predictions) if y_true == y_pred)
    accuracy = correct / len(data)
    f1 = f1_score(labels, predictions)

    print(f"Evaluation -> Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    if output_path:
        export_predictions(output_path, results)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train and evaluate a linear SVM from scratch.")
    parser.add_argument("--data-path", type=Path, default=base_dir / "processed_data.csv", help="Path to normalized dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--C", type=float, default=10.0, help="Regularization strength.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and shuffling.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=base_dir / "test_results_scratch.csv",
        help="Destination CSV for test predictions.",
    )
    parser.add_argument("--skip-export", action="store_true", help="Do not write test predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_processed_data(args.data_path)
    train_data, val_data, test_data = split_data(
        data, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    print("Training SVM...")
    w, b = train_svm(train_data, epochs=args.epochs, lr=args.lr, C=args.C, seed=args.seed)

    print("\nValidation Set:")
    evaluate(val_data, w, b)

    print("\nTest Set:")
    output_path = None if args.skip_export else args.output_path
    evaluate(test_data, w, b, output_path=output_path)


if __name__ == "__main__":
    main()
