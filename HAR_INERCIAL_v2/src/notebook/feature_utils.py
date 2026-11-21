# -*- coding: utf-8 -*-
"""
Pure-Python Inertial Feature Engineering & Classical Machine Learning Suite (HAR_INERCIAL_v2).
Replaces both Octave (MFCC/spectral extraction) and Weka (J48/RandomForest) with fast,
vectorized NumPy/SciPy signal processing and Scikit-Learn pipelines.
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq, dct
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def _create_mel_filterbank(num_filters, n_fft, fs):
    """Creates a triangular Mel-scale filterbank matrix for inertial signals (0 to fs/2 Hz)."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(fs / 2.0)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    num_bins = n_fft // 2 + 1
    fbank = np.zeros((num_filters, num_bins), dtype=np.float32)

    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = max(bin_points[m], f_m_minus + 1)
        f_m_plus = max(bin_points[m + 1], f_m + 1)

        for k in range(f_m_minus, min(f_m, num_bins)):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, min(f_m_plus, num_bins)):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank


def extract_imfcc_features(signal_2d, fs=50.0, n_mfcc=8, n_filters=16):
    """
    Computes Inertial Mel-Frequency Cepstral Coefficients (IMFCC) for a batch of 1D signals
    of shape (N, T) using a Hamming window, FFT power spectrum, Mel filterbank, and DCT-II.
    """
    N, T = signal_2d.shape
    window = np.hamming(T).astype(np.float32)
    windowed = (signal_2d - np.mean(signal_2d, axis=1, keepdims=True)) * window
    mag_spec = np.abs(rfft(windowed, n=T, axis=1))
    pow_spec = (1.0 / T) * (mag_spec ** 2)

    fbank = _create_mel_filterbank(n_filters, T, fs)
    filter_energies = np.dot(pow_spec, fbank.T)
    filter_energies = np.maximum(filter_energies, 1e-10)
    log_energies = np.log(filter_energies)

    mfccs = dct(log_energies, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return mfccs.astype(np.float32)


def extract_inertial_features(df_csv, num_steps=200, fs=50.0, n_mfcc=8):
    """
    Extracts a comprehensive, physically interpretable set of Time-Domain, Frequency-Domain (FFT/PSD),
    and Cepstral (IMFCC) features from 3-axis accelerometer signals:
      - Per-axis (X, Y, Z) and Signal Magnitude Vector (SMV = sqrt(X^2 + Y^2 + Z^2)):
        * Time-domain: mean, std, rms, min, max, peak-to-peak (ptp), IQR, zero-crossing rate (zcr),
          mean absolute jerk, skewness, kurtosis.
        * Frequency-domain: dominant frequency, spectral centroid, spectral entropy,
          low-band energy (0.5-3 Hz), mid-band energy (3-10 Hz), high-band energy (10-25 Hz).
      - Cross-axis kinematics: Signal Magnitude Area (SMA), correlations corr(X,Y), corr(X,Z), corr(Y,Z).
      - Cepstral domain: Top `n_mfcc` Inertial MFCCs per axis (X, Y, Z).

    Returns a DataFrame with engineered features + 'class' column.
    """
    t0 = time.perf_counter()
    raw = df_csv.iloc[:, : 3 * num_steps].values.astype(np.float32)
    X_ax = raw[:, :num_steps]
    Y_ax = raw[:, num_steps : 2 * num_steps]
    Z_ax = raw[:, 2 * num_steps : 3 * num_steps]
    SMV = np.sqrt(X_ax ** 2 + Y_ax ** 2 + Z_ax ** 2)

    freqs = rfftfreq(num_steps, d=1.0 / fs)
    non_dc_freqs = freqs[1:]

    low_mask = (non_dc_freqs >= 0.5) & (non_dc_freqs < 3.0)
    mid_mask = (non_dc_freqs >= 3.0) & (non_dc_freqs < 10.0)
    high_mask = non_dc_freqs >= 10.0

    feat_dict = {}

    for axis_name, sig in [("X", X_ax), ("Y", Y_ax), ("Z", Z_ax), ("SMV", SMV)]:
        mean_val = np.mean(sig, axis=1)
        std_val = np.std(sig, axis=1)
        rms_val = np.sqrt(np.mean(sig ** 2, axis=1))
        min_val = np.min(sig, axis=1)
        max_val = np.max(sig, axis=1)
        ptp_val = max_val - min_val
        q75, q25 = np.percentile(sig, [75, 25], axis=1)
        iqr_val = q75 - q25

        centered = sig - mean_val[:, None]
        zcr_val = np.mean(np.diff(np.signbit(centered), axis=1), axis=1)
        jerk_val = np.mean(np.abs(np.diff(sig, axis=1)), axis=1)
        skew_val = np.nan_to_num(stats.skew(sig, axis=1), nan=0.0)
        kurt_val = np.nan_to_num(stats.kurtosis(sig, axis=1), nan=0.0)

        feat_dict[f"{axis_name}_mean"] = mean_val
        feat_dict[f"{axis_name}_std"] = std_val
        feat_dict[f"{axis_name}_rms"] = rms_val
        feat_dict[f"{axis_name}_min"] = min_val
        feat_dict[f"{axis_name}_max"] = max_val
        feat_dict[f"{axis_name}_ptp"] = ptp_val
        feat_dict[f"{axis_name}_iqr"] = iqr_val
        feat_dict[f"{axis_name}_zcr"] = zcr_val
        feat_dict[f"{axis_name}_jerk"] = jerk_val
        feat_dict[f"{axis_name}_skew"] = skew_val
        feat_dict[f"{axis_name}_kurt"] = kurt_val

        # Spectral features (excluding DC bin 0)
        fft_mag = np.abs(rfft(centered, n=num_steps, axis=1))[:, 1:]
        psd = (fft_mag ** 2) / num_steps
        psd_sum = np.sum(psd, axis=1, keepdims=True) + 1e-10
        psd_norm = psd / psd_sum

        dom_idx = np.argmax(psd, axis=1)
        dom_freq = non_dc_freqs[dom_idx]
        spec_centroid = np.sum(psd_norm * non_dc_freqs[None, :], axis=1)
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1)

        feat_dict[f"{axis_name}_dom_freq"] = dom_freq
        feat_dict[f"{axis_name}_spec_centroid"] = spec_centroid
        feat_dict[f"{axis_name}_spec_entropy"] = spec_entropy
        feat_dict[f"{axis_name}_energy_low"] = np.sum(psd[:, low_mask], axis=1)
        feat_dict[f"{axis_name}_energy_mid"] = np.sum(psd[:, mid_mask], axis=1)
        feat_dict[f"{axis_name}_energy_high"] = np.sum(psd[:, high_mask], axis=1)

    # Cross-axis features: Signal Magnitude Area (SMA) and pairwise axis correlations
    feat_dict["SMA"] = np.mean(np.abs(X_ax) + np.abs(Y_ax) + np.abs(Z_ax), axis=1)

    def _batch_corr(a, b):
        a_c = a - np.mean(a, axis=1, keepdims=True)
        b_c = b - np.mean(b, axis=1, keepdims=True)
        denom = np.sqrt(np.sum(a_c ** 2, axis=1) * np.sum(b_c ** 2, axis=1)) + 1e-10
        return np.sum(a_c * b_c, axis=1) / denom

    feat_dict["corr_XY"] = _batch_corr(X_ax, Y_ax)
    feat_dict["corr_XZ"] = _batch_corr(X_ax, Z_ax)
    feat_dict["corr_YZ"] = _batch_corr(Y_ax, Z_ax)

    # Cepstral features (IMFCC) per physical axis
    if n_mfcc > 0:
        for axis_name, sig in [("X", X_ax), ("Y", Y_ax), ("Z", Z_ax)]:
            imfccs = extract_imfcc_features(sig, fs=fs, n_mfcc=n_mfcc)
            for c_idx in range(n_mfcc):
                feat_dict[f"{axis_name}_imfcc_{c_idx + 1}"] = imfccs[:, c_idx]

    df_features = pd.DataFrame(feat_dict)
    df_features["class"] = df_csv["class"].values

    elapsed = time.perf_counter() - t0
    print(
        f"[Feature Extraction] Extracted {df_features.shape[1] - 1} engineered features "
        f"for {len(df_features)} windows in {elapsed:.3f} s"
    )
    return df_features


def train_and_evaluate_sklearn(
    df_train,
    df_test,
    classifier_type="DecisionTree",
    random_state=42,
    verbose=True,
    **kwargs,
):
    """
    Trains and evaluates a classical Scikit-Learn classifier on either raw inertial data
    or engineered features.
    Supported classifier_type values:
      - 'DecisionTree' (CART / Weka J48 equivalent)
      - 'RandomForest'
      - 'GradientBoosting'
      - 'SVM_RBF'
    """
    drop_cols = [c for c in ["class", "user"] if c in df_train.columns]
    X_train = df_train.drop(columns=drop_cols).values
    y_train = df_train["class"].values

    drop_cols_test = [c for c in ["class", "user"] if c in df_test.columns]
    X_test = df_test.drop(columns=drop_cols_test).values
    y_test = df_test["class"].values

    if classifier_type == "DecisionTree":
        clf = DecisionTreeClassifier(
            criterion=kwargs.get("criterion", "entropy"),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            random_state=random_state,
        )
    elif classifier_type == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            max_depth=kwargs.get("max_depth", None),
            random_state=random_state,
            n_jobs=-1,
        )
    elif classifier_type == "GradientBoosting":
        clf = GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            random_state=random_state,
        )
    elif classifier_type == "SVM_RBF":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=kwargs.get("C", 10.0), kernel="rbf", probability=True, random_state=random_state)),
        ])
    else:
        raise ValueError(
            f"Unknown classifier_type '{classifier_type}'. Choose from: "
            "'DecisionTree', 'RandomForest', 'GradientBoosting', 'SVM_RBF'."
        )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = 100.0 * accuracy_score(y_train, y_pred_train)
    test_acc = 100.0 * accuracy_score(y_test, y_pred_test)
    test_f1 = 100.0 * f1_score(y_test, y_pred_test, average="macro")

    if verbose:
        print(f"=== [{classifier_type}] Evaluation Summary ===")
        print(f"  Training Time : {train_time:.3f} s")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy : {test_acc:.2f}% | Macro F1: {test_f1:.2f}%\n")
        print("Confusion Matrix (Test Set):")
        print(confusion_matrix(y_test, y_pred_test))
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred_test, digits=4))

    return {
        "model": clf,
        "classifier_type": classifier_type,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "train_time": train_time,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_test": y_test,
    }
