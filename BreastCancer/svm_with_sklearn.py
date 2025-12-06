import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import export_predictions, load_processed_data, split_data

Example = Tuple[str, List[float], int]


def prepare_features_labels(data: Sequence[Example], num_features: Optional[int] = None):
    ids = [id_val for id_val, _, _ in data]
    X = [features[:num_features] if num_features else features for _, features, _ in data]
    y = [label for _, _, label in data]
    return ids, np.array(X), np.array(y)


def evaluate_model(
    ids: Sequence[str], y_true: Sequence[int], y_pred: Sequence[int], output_path: Optional[Path] = None
):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Evaluation -> Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    if output_path:
        export_predictions(output_path, zip(ids, y_true, y_pred))
    return acc, f1


def plot_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], save_path: Optional[Path] = None, show: bool = False
):
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["1 (Malignant)", "-1 (Benign)"],
        yticklabels=["1 (Malignant)", "-1 (Benign)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_decision_boundary(
    model: SVC, X: np.ndarray, y: np.ndarray, title: str, save_path: Optional[Path] = None, show: bool = False
):
    step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)

    class_1 = y == 1
    class_neg1 = y == -1
    plt.scatter(X[class_1, 0], X[class_1, 1], c="red", label="Malignant (1)", edgecolors="k")
    plt.scatter(X[class_neg1, 0], X[class_neg1, 1], c="blue", label="Benign (-1)", edgecolors="k")

    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Decision boundary saved to: {save_path}")

    if show:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train and evaluate an SVM using scikit-learn.")
    parser.add_argument("--data-path", type=Path, default=base_dir / "processed_data.csv", help="Path to normalized dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and PCA.")
    parser.add_argument("--kernel", default="linear", help="SVC kernel.")
    parser.add_argument("--C", type=float, default=10.0, help="Regularization strength.")
    parser.add_argument("--gamma", default="scale", help="Kernel coefficient for rbf/poly/sigmoid.")
    parser.add_argument("--degree", type=int, default=3, help="Degree for the poly kernel.")
    parser.add_argument(
        "--output-path", type=Path, default=base_dir / "test_results_sklearn.csv", help="Destination CSV for test predictions."
    )
    parser.add_argument("--skip-export", action="store_true", help="Do not write test predictions.")
    parser.add_argument("--confusion-path", type=Path, default=base_dir / "images/confusion_matrix.png")
    parser.add_argument("--decision-path", type=Path, default=base_dir / "images/decision_boundary_pca.png")
    parser.add_argument("--show-plots", action="store_true", help="Display matplotlib windows.")
    parser.add_argument("--skip-plots", action="store_true", help="Disable all plotting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    data = load_processed_data(args.data_path)
    train_data, val_data, test_data = split_data(
        data, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    train_ids, X_train, y_train = prepare_features_labels(train_data)
    val_ids, X_val, y_val = prepare_features_labels(val_data)
    test_ids, X_test, y_test = prepare_features_labels(test_data)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        degree=args.degree,
        probability=False,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    print("\nTrain Set:")
    evaluate_model(train_ids, y_train, model.predict(X_train))

    print("\nValidation Set:")
    evaluate_model(val_ids, y_val, model.predict(X_val))

    print("\nTest Set:")
    test_output = None if args.skip_export else args.output_path
    test_preds = model.predict(X_test)
    evaluate_model(test_ids, y_test, test_preds, output_path=test_output)

    if not args.skip_plots:
        plot_confusion_matrix(y_test, test_preds, save_path=args.confusion_path, show=args.show_plots)

        print("\nApplying PCA for visualization...")
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        pca = PCA(n_components=2, random_state=args.seed)
        X_all_pca = pca.fit_transform(X_all)

        model_pca = SVC(kernel="linear", C=args.C, gamma=args.gamma)
        model_pca.fit(X_all_pca, y_all)
        plot_decision_boundary(
            model_pca,
            X_all_pca,
            y_all,
            title="Decision Boundary (Train + Validation PCA 2D)",
            save_path=args.decision_path,
            show=args.show_plots,
        )


if __name__ == "__main__":
    main()
