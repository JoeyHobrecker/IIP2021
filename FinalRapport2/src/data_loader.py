"""Utility for loading EMG+IMU CSV datasets."""
from pathlib import Path
from typing import Tuple, List

import pandas as pd

CSV_COLUMNS = [
    "timestamp",
    "emg_ch1",
    # Add additional EMG channels as needed
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_gx",
    "imu_gy",
    "imu_gz",
    "label",
    "subject_id",
]


def load_dataset(data_folder: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load all subject CSV files from a folder.

    Expected CSV schema:
    timestamp, emg_ch1, ..., imu_gz, label?, subject_id?
    """
    path = Path(data_folder)
    frames: List[pd.DataFrame] = []
    for csv_file in sorted(path.glob("subject_*_raw.csv")):
        subject_id = csv_file.stem.split("_")[1]
        df = pd.read_csv(csv_file, header=None)
        n_cols = df.shape[1]
        expected = len(CSV_COLUMNS) - 2  # label and subject_id may not exist
        if n_cols < expected:
            # TODO: handle malformed CSVs
            continue
        df.columns = CSV_COLUMNS[:n_cols]
        if "label" not in df.columns:
            df["label"] = "unlabeled"
        df["subject_id"] = subject_id
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CSV_COLUMNS)
    X_raw = all_df.drop(["label", "subject_id"], axis=1)
    y_labels = all_df["label"]
    subject_ids = all_df["subject_id"]
    return X_raw, y_labels, subject_ids
