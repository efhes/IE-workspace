# -*- coding: utf-8 -*-
"""
Data loading, preprocessing, and statistical utilities for HAR_INERCIAL_v2.
100% pure Python (NumPy / Pandas / Scikit-Learn) with zero Weka/Octave dependencies.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_PINGPONG_MAP = {
    1: "DRIVE", 2: "BACKHAND", 3: "SERVE", 4: "LOB",
    "1": "DRIVE", "2": "BACKHAND", "3": "SERVE", "4": "LOB",
}


def DownloadDS(filename, csv_with_user_info=False):
    """
    Loads an inertial CSV dataset (600 accelerometer readings + class [+ optional user]).
    Automatically detects:
      - Whether the first line is a header ('param1,...') or raw data.
      - Whether the file has 601 columns (no user) or 602 columns (with user).
      - Incomplete/corrupted rows containing NaN values (automatically dropped with a warning).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    first_tokens = first_line.split(",")
    has_header = first_tokens[0].lower().startswith("param")

    df_csv = pd.read_csv(
        filename,
        sep=",",
        header=0 if has_header else None,
        low_memory=False,
    )

    num_cols = df_csv.shape[1]
    param_cols = [f"param{i}" for i in range(1, 601)]

    if num_cols == 602:
        df_csv.columns = param_cols + ["class", "user"]
    elif num_cols == 601:
        df_csv.columns = param_cols + ["class"]
    else:
        raise ValueError(
            f"Unexpected number of columns ({num_cols}) in {filename}. Expected 601 or 602."
        )

    # Coerce accelerometer columns to numeric and drop any malformed/partially-synced rows
    for c in param_cols:
        if not pd.api.types.is_numeric_dtype(df_csv[c]):
            df_csv[c] = pd.to_numeric(df_csv[c], errors="coerce")

    if df_csv.isna().any().any():
        nan_rows = int(df_csv.isna().any(axis=1).sum())
        print(f"[DownloadDS][WARNING] Dropping {nan_rows} incomplete row(s) containing NaN values...")
        df_csv = df_csv.dropna().reset_index(drop=True)

    df_csv["class"] = df_csv["class"].astype(str).str.strip()
    if set(df_csv["class"].unique()).issubset({"1", "2", "3", "4"}):
        df_csv["class"] = df_csv["class"].map(DEFAULT_PINGPONG_MAP)

    if not csv_with_user_info and "user" in df_csv.columns:
        df_csv = df_csv.drop(columns=["user"])

    return df_csv


def load_and_split_dataset(
    filepath,
    percentage_test_split=0.2,
    random_state=42,
    csv_with_user_info=False,
):
    """
    Loads the inertial CSV dataset, determines the sorted classes and class mapping,
    and performs a stratified train/test split.
    """
    df_csv = DownloadDS(filepath, csv_with_user_info=csv_with_user_info)

    drop_cols = ["class", "user"] if "user" in df_csv.columns else ["class"]
    X = df_csv.drop(columns=drop_cols)
    y = df_csv["class"]

    classes_list = sorted(y.unique().tolist())
    num_classes = len(classes_list)
    class_mapping = {i + 1: name for i, name in enumerate(classes_list)}
    root_class_list = "_".join(classes_list)

    stratify_y = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=percentage_test_split,
        random_state=random_state,
        stratify=stratify_y,
    )

    df_train_csv = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    df_test_csv = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    print(f"[DATASET LOADED] {filepath}")
    print(f"  Total samples : {len(df_csv)} ({df_train_csv.shape[0]} train / {df_test_csv.shape[0]} test)")
    print(f"  Signal format : 600 features (200 timesteps @ 50 Hz x 3 axes [X, Y, Z])")
    print(f"  Classes ({num_classes})   : {classes_list}")
    print(f"  Class mapping : {class_mapping}")

    return (
        df_csv,
        df_train_csv,
        df_test_csv,
        classes_list,
        num_classes,
        class_mapping,
        root_class_list,
    )


def PrepareData(df_csv, num_steps=200, dimensions=3, classes_list=None, verbose=True):
    """
    Vectorized preparation of 3D sequential tensors for Recurrent / 1D-CNN models in Keras 3.
    Converts flat 600-column rows [AccX(1..200), AccY(1..200), AccZ(1..200)] into
    3D tensor X of shape (num_samples, num_steps=200, dimensions=3) and one-hot labels y
    of shape (num_samples, num_classes).
    """
    if "user" in df_csv.columns:
        df_csv = df_csv.drop(columns=["user"])

    if df_csv.isna().any().any():
        nan_rows = int(df_csv.isna().any(axis=1).sum())
        print(f"[PrepareData][WARNING] Dropping {nan_rows} incomplete row(s) containing NaN values...")
        df_csv = df_csv.dropna().reset_index(drop=True)

    X_pre = df_csv.iloc[:, : num_steps * dimensions].values.astype(np.float32)
    y_pre = df_csv.iloc[:, -1].astype(str)

    if classes_list is not None:
        ordered_classes = sorted(classes_list)
        y_cat = pd.Categorical(y_pre, categories=ordered_classes, ordered=True)
        y = pd.get_dummies(y_cat, dtype=np.float32).values
    else:
        y = pd.get_dummies(y_pre, dtype=np.float32).values

    num_samples = X_pre.shape[0]
    # Vectorized reshape: (N, 3, 200) -> transpose to (N, 200, 3)
    # Identical to stacking [X_pre[j, i], X_pre[j, i+200], X_pre[j, i+400]] for each timestep i
    X = X_pre.reshape(num_samples, dimensions, num_steps).transpose(0, 2, 1).copy()

    if verbose:
        print(f"[PrepareData] Samples: {num_samples} | X shape: {X.shape} (dtype={X.dtype}) | y shape: {y.shape}")

    return X, y


def GetAccelerationAxis(df_csv, num_steps=200):
    """
    Splits the dataframe into X, Y, Z axis DataFrames, full 600-param DataFrame, and class Series.
    """
    AccX = df_csv.loc[:, "param1":f"param{num_steps}"]
    AccY = df_csv.loc[:, f"param{num_steps + 1}":f"param{2 * num_steps}"]
    AccZ = df_csv.loc[:, f"param{2 * num_steps + 1}":f"param{3 * num_steps}"]
    AccAll = df_csv.loc[:, "param1":f"param{3 * num_steps}"]
    y = df_csv["class"]
    return AccX, AccY, AccZ, AccAll, y


def CalculateCI(num_samples, accuracy):
    """
    Calculates 95% confidence interval half-width for a given accuracy percentage (0-100).
    """
    accuracy = float(np.clip(accuracy, 0.0, 100.0))
    return float(1.96 * np.sqrt(((100.0 - accuracy) * accuracy) / max(int(num_samples), 1)))
