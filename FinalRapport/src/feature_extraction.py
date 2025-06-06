import numpy as np
import pandas as pd

# --- copied from 8_2_Camera_Based_Activity_Recognition/t5_LeapMotion/SaveARFF_Postures/SaveARFF_Postures.pde: original lines 85-92 ---
# float[] appendArrayTail (float[] _array, float _val) {
#   float[] array = _array;
#   float[] tempArray = new float[_array.length-1];
#   arrayCopy(array, 1, tempArray, 0, tempArray.length);
#   array[tempArray.length] = _val;
#   arrayCopy(tempArray, 0, array, 0, tempArray.length);
#   return array;
# }

# --- copied from 4_2_Real_Time_Motion_Classification/Processing/1_Classification/e4_2a_A0TrainLSVC/e4_2a_A0TrainLSVC.pde: original lines 117-125 ---
# float[] appendArray (float[] _array, float _val) {
#   float[] array = _array;
#   float[] tempArray = new float[_array.length-1];
#   arrayCopy(array, tempArray, tempArray.length);
#   array[0] = _val;
#   arrayCopy(tempArray, 0, array, 1, tempArray.length);
#   return array;
# }


def sliding_window(data: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """Generate sliding windows with given size and step."""
    num_samples = data.shape[0]
    windows = []
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)


def extract_emg_features(window: np.ndarray) -> np.ndarray:
    """Placeholder for EMG feature extraction (RMS, MAV, ZC, etc.)."""
    # TODO: replace with real EMG feature extraction
    rms = np.sqrt(np.mean(np.square(window), axis=0))
    mav = np.mean(np.abs(window), axis=0)
    return np.concatenate([rms, mav])


def extract_imu_features(window: np.ndarray) -> np.ndarray:
    """Placeholder for IMU feature extraction."""
    # TODO: replace with real IMU feature extraction
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    return np.concatenate([mean, std])


def balance_classes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simple random undersampling to balance classes."""
    uniq, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    indices = np.hstack([np.random.choice(np.where(y == u)[0], min_count, replace=False) for u in uniq])
    return X[indices], y[indices]
