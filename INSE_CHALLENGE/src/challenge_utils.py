#!/usr/bin/env python3
"""
challenge_utils.py
------------------
Lightweight utility functions for students and researchers to easily load,
reshape, and prepare the tennis HAR challenge dataset.
Supports automatic engine detection and graceful fallback between Parquet, CSV, and NPZ.
"""

import os
from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd

VALID_CLASSES = ['BackhandRH', 'BackhandTwoH', 'Drive', 'Lob', 'Serve', 'Waiting']
SENSORS = ['chest', 'waist', 'wrist_L', 'wrist_R']
AXES = ['x', 'y', 'z']
WINDOW_SIZE = 200
N_CHANNELS = 12

CHANNEL_NAMES = [f"{sensor}_{axis}" for sensor in SENSORS for axis in AXES]
FEATURE_COLUMNS = [f"{ch}_t{t}" for t in range(WINDOW_SIZE) for ch in CHANNEL_NAMES]
FUNCTIONAL_NAMES = ['mean', 'std', 'var', 'min', 'max', 'median']
FUNCTIONAL_FEATURE_NAMES = [f"{ch}_{stat}" for stat in FUNCTIONAL_NAMES for ch in CHANNEL_NAMES]


def has_parquet_engine() -> bool:
    """Checks if a parquet engine (pyarrow or fastparquet) is installed."""
    try:
        import pyarrow
        return True
    except ImportError:
        try:
            import fastparquet
            return True
        except ImportError:
            return False


def _read_challenge_df(data_dir: str, fname_base: str, prefer_format: str = "auto") -> pd.DataFrame:
    """
    Reads a challenge dataset as a DataFrame, automatically selecting the best available format
    (Parquet, CSV) with graceful fallback if pyarrow/fastparquet is not installed.
    """
    parquet_path = os.path.join(data_dir, f"{fname_base}.parquet")
    csv_path = os.path.join(data_dir, f"{fname_base}.csv")

    can_use_parquet = has_parquet_engine() and os.path.exists(parquet_path)

    if prefer_format in ("auto", "parquet") and can_use_parquet:
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ValueError):
            pass  # Fall back to CSV

    # Fallback to CSV
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # If only parquet exists but engine is missing, attempt read to trigger informative error
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(f"Neither {fname_base}.parquet nor {fname_base}.csv found in {data_dir}")


def to_tensor(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts the 2,400 sensor feature columns from a challenge DataFrame
    and reshapes them into a 3D NumPy array of shape (N, 200, 12).

    Parameters:
        df (pd.DataFrame): DataFrame containing feature columns 'chest_x_t0' .. 'wrist_R_z_t199'.

    Returns:
        X (np.ndarray): Tensor of shape (N, 200, 12) with dtype float32.
    """
    features = df[FEATURE_COLUMNS].values.astype(np.float32)
    return features.reshape(-1, WINDOW_SIZE, N_CHANNELS)


def compute_functionals(X: np.ndarray) -> np.ndarray:
    """
    Computes 6 statistical functional descriptors (mean, std, var, min, max, median)
    across the temporal window axis (axis=1) for all 12 channels.

    Parameters:
        X (np.ndarray): Tensor of shape (N, 200, 12).

    Returns:
        feats (np.ndarray): Array of shape (N, 72) with 72 extracted functional features:
                            [mean(12), std(12), var(12), min(12), max(12), median(12)].
    """
    if X.ndim != 3 or X.shape[1] != WINDOW_SIZE or X.shape[2] != N_CHANNELS:
        raise ValueError(f"Expected tensor of shape (N, {WINDOW_SIZE}, {N_CHANNELS}), got {X.shape}")

    mean_v = np.mean(X, axis=1)
    std_v = np.std(X, axis=1)
    var_v = np.var(X, axis=1)
    min_v = np.min(X, axis=1)
    max_v = np.max(X, axis=1)
    med_v = np.median(X, axis=1)
    return np.hstack([mean_v, std_v, var_v, min_v, max_v, med_v])


def load_train(data_dir: str = "./challenge_data",
               prefer_format: str = "auto") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads the training dataset.

    Parameters:
        data_dir (str): Directory containing the challenge files.
        prefer_format (str): Preferred format ('auto', 'parquet', 'npz', or 'csv').

    Returns:
        X (np.ndarray): Shape (N, 200, 12) acceleration tensor.
        y (np.ndarray): Shape (N,) string array with gesture labels.
        meta (pd.DataFrame): DataFrame containing metadata ('sample_id', 'subject_id',
                             'device_type', 'session', 'timestamp').
    """
    npz_path = os.path.join(data_dir, "train.npz")
    if prefer_format == "npz" and os.path.exists(npz_path):
        data = np.load(npz_path)
        meta = pd.DataFrame({
            'sample_id': data['sample_id'],
            'subject_id': data['subject_id'],
            'device_type': data['device_type'],
            'session': data['session'],
            'timestamp': data['timestamp'],
            'label': data['y']
        })
        return data['X'], data['y'], meta

    df = _read_challenge_df(data_dir, "train", prefer_format=prefer_format)
    meta_cols = ['sample_id', 'subject_id', 'device_type', 'session', 'timestamp']
    meta = df[meta_cols]
    y = df['label'].values
    X = to_tensor(df)
    return X, y, meta


def load_test(data_dir: str = "./challenge_data",
              labeled: bool = False,
              prefer_format: str = "auto") -> Union[Tuple[np.ndarray, pd.DataFrame],
                                                   Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    Loads the test dataset.

    Parameters:
        data_dir (str): Directory containing the challenge files.
        labeled (bool): If True, loads 'test_labeled' with ground-truth labels.
                        If False (default), loads 'test_unlabeled' (blind challenge set).
        prefer_format (str): Preferred format ('auto', 'parquet', 'npz', or 'csv').

    Returns:
        If labeled=False:
            X (np.ndarray): Shape (N, 200, 12) acceleration tensor.
            meta (pd.DataFrame): Metadata ('sample_id', 'subject_id', 'device_type', 'session', 'timestamp').
        If labeled=True:
            X (np.ndarray): Shape (N, 200, 12) acceleration tensor.
            y (np.ndarray): Shape (N,) ground-truth string labels.
            meta (pd.DataFrame): Metadata DataFrame.
    """
    fname_base = "test_labeled" if labeled else "test_unlabeled"
    npz_path = os.path.join(data_dir, f"{fname_base}.npz")

    if prefer_format == "npz" and os.path.exists(npz_path):
        data = np.load(npz_path)
        meta_dict = {
            'sample_id': data['sample_id'],
            'subject_id': data['subject_id'],
            'device_type': data['device_type'],
            'session': data['session'],
            'timestamp': data['timestamp'],
        }
        if labeled:
            meta_dict['label'] = data['y']
            return data['X'], data['y'], pd.DataFrame(meta_dict)
        return data['X'], pd.DataFrame(meta_dict)

    df = _read_challenge_df(data_dir, fname_base, prefer_format=prefer_format)
    meta_cols = ['sample_id', 'subject_id', 'device_type', 'session', 'timestamp']
    meta = df[meta_cols]
    X = to_tensor(df)

    if labeled:
        y = df['label'].values
        return X, y, meta
    return X, meta


def create_submission(sample_ids, predictions, output_file: str = "student_submission.csv"):
    """
    Creates and validates a student submission file.

    Parameters:
        sample_ids: List or array of sample IDs (e.g. 'test_00001').
        predictions: List or array of predicted gesture labels.
        output_file (str): Destination CSV path.
    """
    if len(sample_ids) != len(predictions):
        raise ValueError(f"Length mismatch: {len(sample_ids)} sample_ids vs {len(predictions)} predictions.")

    invalid = set(predictions) - set(VALID_CLASSES)
    if invalid:
        raise ValueError(f"Invalid gesture labels in predictions: {invalid}. Must be in {VALID_CLASSES}")

    df_sub = pd.DataFrame({
        'sample_id': sample_ids,
        'predicted_label': predictions
    })
    df_sub.to_csv(output_file, index=False)
    print(f"[Submission Saved] {len(df_sub)} predictions successfully written to: {output_file}")
