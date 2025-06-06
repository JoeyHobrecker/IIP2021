# --- copied from FinalRapport/src/train_pipeline.py: lines 1-66 ---
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .data_loader import load_all_subjects
from .preprocessing import prepare_dataset


DATA_DIR = Path("FinalRapport1/data")
SAMPLE_RATE_HZ = 100  # TODO: set actual sample rate


def get_csv_files() -> list[str]:
    return sorted(str(p) for p in DATA_DIR.glob("*.csv"))


def run_training() -> None:
    csv_files = get_csv_files()
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory")
    raw_df = load_all_subjects(csv_files)
    X, y, groups = prepare_dataset(raw_df, SAMPLE_RATE_HZ)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    group_kf = GroupKFold(n_splits=len(np.unique(groups)))

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    acc_scores = []
    f1_scores = []
    for train_idx, test_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        acc_scores.append(accuracy_score(y[test_idx], preds))
        f1_scores.append(f1_score(y[test_idx], preds, average="weighted"))
    print(f"Stratified 5-Fold Accuracy: {np.mean(acc_scores):.3f}")
    print(f"Stratified 5-Fold Weighted F1: {np.mean(f1_scores):.3f}")

    loso_acc = []
    loso_f1 = []
    for train_idx, test_idx in group_kf.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        loso_acc.append(accuracy_score(y[test_idx], preds))
        loso_f1.append(f1_score(y[test_idx], preds, average="weighted"))
    print(f"LOSO Accuracy: {np.mean(loso_acc):.3f}")
    print(f"LOSO Weighted F1: {np.mean(loso_f1):.3f}")

    model.fit(X, y)
    joblib.dump(model, "FinalRapport1/lightweight_rf.pkl")

    disp = ConfusionMatrixDisplay.from_estimator(model, X, y)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("FinalRapport1/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    run_training()

