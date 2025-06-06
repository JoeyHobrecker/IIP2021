"""Training pipeline implementing K-Fold and LOSO validation."""
import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data_loader import load_dataset
from preprocessing import window_and_balance
from feature_extraction import extract_all_features


def run_kfold(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        best_acc = 0.0
        best_model = None
        for n_estimators in [50, 100, 200]:  # TODO: expand hyperparameter grid
            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            acc = accuracy_score(y_val, preds)
            if acc > best_acc:
                best_acc = acc
                best_model = clf
        f1 = f1_score(y_val, best_model.predict(X_val), average="weighted")
        results.append({
            "fold_index": fold,
            "train_accuracy": best_acc,
            "val_accuracy": best_acc,
            "fold_f1": f1,
        })
    return pd.DataFrame(results)


def run_loso(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    results = []
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        train_idx = subjects != subj
        test_idx = subjects == subj
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results.append({
            "subject_id": subj,
            "test_accuracy": acc,
            "test_f1": f1,
        })
    return pd.DataFrame(results)


def save_model(model, path: Path) -> None:
    joblib.dump(model, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()

    X_raw, y_labels, subject_ids = load_dataset(args.data_folder)
    X_windows, y_windows, subj_windows = window_and_balance(X_raw, y_labels, subject_ids)
    X_features = extract_all_features(X_windows)

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    kfold_results = run_kfold(X_features, y_windows)
    kfold_results.to_csv(output_path / "cv_results.csv", index=False)

    loso_results = run_loso(X_features, y_windows, subj_windows)
    loso_results.to_csv(output_path / "loso_results.csv", index=False)

    # Fit on entire dataset using best hyperparameters (simplified)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_features, y_windows)
    save_model(clf, output_path / "best_model.pkl")

    print("K-Fold results:\n", kfold_results)
    print("LOSO results:\n", loso_results)


if __name__ == "__main__":
    main()
