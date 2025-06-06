import numpy as np
import pandas as pd
from typing import Tuple

from .feature_extraction import sliding_window, extract_emg_features, extract_imu_features, balance_classes

WINDOW_MS = 200
OVERLAP = 0.5


def prepare_dataset(df: pd.DataFrame, sample_rate_hz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window the raw dataframe and extract features."""
    window_size = int(sample_rate_hz * WINDOW_MS / 1000)
    step_size = int(window_size * (1 - OVERLAP))
    emg_cols = [c for c in df.columns if "emg" in c.lower()]
    imu_cols = [c for c in df.columns if "imu" in c.lower()]
    X_list = []
    y_list = []
    groups = []
    for subject, sub_df in df.groupby("subject"):
        emg_data = sub_df[emg_cols].to_numpy()
        imu_data = sub_df[imu_cols].to_numpy()
        labels = sub_df["label"].to_numpy()
        windows = sliding_window(np.arange(len(sub_df)), window_size, step_size)
        for w in windows:
            emg_window = emg_data[w]
            imu_window = imu_data[w]
            label = np.bincount(labels[w]).argmax()
            features = np.concatenate([
                extract_emg_features(emg_window),
                extract_imu_features(imu_window),
            ])
            X_list.append(features)
            y_list.append(label)
            groups.append(subject)
    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(groups)
    X, y = balance_classes(X, y)
    return X, y, groups

