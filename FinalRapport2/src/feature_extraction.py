"""Feature extraction for EMG and IMU windows."""
from typing import List

import numpy as np


# These features are suggestions; user should verify correct implementation for ADS1299 channel arrangement.

def extract_emg_features(window_emg: np.ndarray) -> List[float]:
    """Extract EMG features from a single window."""
    rms = np.sqrt(np.mean(window_emg ** 2))
    mav = np.mean(np.abs(window_emg))
    zero_crossings = np.sum(np.diff(np.signbit(window_emg)))
    ptp_amp = np.ptp(window_emg)
    return [rms, mav, zero_crossings, ptp_amp]


def extract_imu_features(window_imu: np.ndarray) -> List[float]:
    """Extract IMU features from a single window."""
    imu_mean = np.mean(window_imu, axis=0)
    imu_var = np.var(window_imu, axis=0)
    features = np.hstack([imu_mean, imu_var])
    # TODO: add tempo features such as time-under-tension
    return features.tolist()


def combine_features(X_emg_windows: np.ndarray, X_imu_windows: np.ndarray) -> np.ndarray:
    """Concatenate EMG and IMU feature vectors."""
    feats = []
    for emg_w, imu_w in zip(X_emg_windows, X_imu_windows):
        emg_feats = extract_emg_features(emg_w)
        imu_feats = extract_imu_features(imu_w)
        feats.append(emg_feats + imu_feats)
    return np.array(feats)


def extract_all_features(X_windows_raw: np.ndarray) -> np.ndarray:
    """High-level feature extraction for all windows.

    Parameters
    ----------
    X_windows_raw : np.ndarray
        Array of shape (num_windows, samples_per_window * num_channels)
    """
    # Assuming first portion is EMG and second is IMU based on window sizes
    # Users should adapt slicing depending on channel order
    num_emg_samples = 1600  # 200 ms * 8 kHz
    num_imu_samples = 20    # 200 ms * 100 Hz
    emg_len = num_emg_samples
    imu_len = num_imu_samples * 6  # 6 axes

    X_emg = X_windows_raw[:, :emg_len]
    X_imu = X_windows_raw[:, emg_len:emg_len + imu_len].reshape(-1, num_imu_samples, 6)

    return combine_features(X_emg, X_imu)
