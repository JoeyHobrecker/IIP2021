"""Preprocessing utilities for EMG+IMU signals."""
from typing import Tuple

import numpy as np
import pandas as pd


def window_and_balance(
    X_raw: pd.DataFrame,
    y: pd.Series,
    subject_ids: pd.Series,
    window_ms: int = 200,
    overlap_pct: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window the raw signals and balance the dataset.

    Parameters
    ----------
    X_raw : pd.DataFrame
        Raw samples with columns matching `data_loader.CSV_COLUMNS` without label.
    y : pd.Series
        Labels corresponding to each sample.
    subject_ids : pd.Series
        Subject identifier per sample.
    window_ms : int, optional
        Length of each window in milliseconds, by default 200 ms.
    overlap_pct : float, optional
        Overlap between windows (0-1), by default 0.5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Windowed data, labels and subject ids.
    """
    sample_rate_emg = 8000  # 8 kHz
    sample_rate_imu = 100   # 100 Hz

    win_samples_emg = int(window_ms * sample_rate_emg / 1000)
    step_emg = int(win_samples_emg * (1 - overlap_pct))
    win_samples_imu = int(window_ms * sample_rate_imu / 1000)
    step_imu = int(win_samples_imu * (1 - overlap_pct))

    emg_cols = [c for c in X_raw.columns if c.startswith("emg")] 
    imu_cols = [c for c in X_raw.columns if c.startswith("imu")]

    windows = []
    labels = []
    subjects = []

    start = 0
    while start + win_samples_emg <= len(X_raw):
        end_emg = start + win_samples_emg
        end_imu = start + win_samples_imu
        win_emg = X_raw[emg_cols].iloc[start:end_emg].to_numpy()
        win_imu = X_raw[imu_cols].iloc[start:end_imu].to_numpy()
        window = np.hstack([win_emg.flatten(), win_imu.flatten()])
        windows.append(window)
        labels.append(y.iloc[start:end_emg].mode()[0])  # TODO: majority label
        subjects.append(subject_ids.iloc[start])
        start += step_emg

    X_windows = np.array(windows)
    y_windows = np.array(labels)
    subject_windows = np.array(subjects)

    # Simple class balancing via undersampling
    classes, counts = np.unique(y_windows, return_counts=True)
    if len(classes) > 1:
        min_count = counts.min()
        keep_idx = np.hstack([
            np.random.choice(np.where(y_windows == cls)[0], min_count, replace=False)
            for cls in classes
        ])
        X_windows = X_windows[keep_idx]
        y_windows = y_windows[keep_idx]
        subject_windows = subject_windows[keep_idx]

    return X_windows, y_windows, subject_windows

# TODO: add function to detect majority label within a window if labels change mid-window
