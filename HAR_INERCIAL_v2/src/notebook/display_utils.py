# -*- coding: utf-8 -*-
"""
Visualization and plotting utilities for HAR_INERCIAL_v2.
Supports any number of gesture classes and modern Matplotlib/Seaborn/Scikit-Learn APIs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import rfft, rfftfreq
from sklearn import decomposition
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from .har_utils import CalculateCI, GetAccelerationAxis
from .feature_utils import extract_imfcc_features


def create_class_mapping(class_names):
    """Creates 1-based dictionary mapping class index -> class name."""
    return {i + 1: class_name for i, class_name in enumerate(class_names)}


def Convert2nominal_activities(class_number, class_mapping):
    """Maps 1-based integer class index to its nominal string label."""
    return class_mapping.get(class_number, "UNK")


def plot_sample(sample2plot, df_csv, class_mapping=None, num_steps=200, fs=50.0):
    """
    Plots the 3-axis accelerometer waveform (X in red, Y in green, Z in blue) for a single window.
    """
    if sample2plot >= len(df_csv):
        sample2plot = 0

    AccX = df_csv.loc[sample2plot, "param1":f"param{num_steps}"].values.astype(float)
    AccY = df_csv.loc[sample2plot, f"param{num_steps + 1}":f"param{2 * num_steps}"].values.astype(float)
    AccZ = df_csv.loc[sample2plot, f"param{2 * num_steps + 1}":f"param{3 * num_steps}"].values.astype(float)
    t = np.linspace(0, num_steps / fs, num_steps)

    raw_cls = df_csv.loc[sample2plot, "class"]
    if isinstance(raw_cls, (int, np.integer)) and class_mapping:
        cls_label = Convert2nominal_activities(int(raw_cls), class_mapping)
    else:
        cls_label = str(raw_cls)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, AccX, color="#d62728", label="Acc X", linewidth=1.5)
    ax.plot(t, AccY, color="#2ca02c", label="Acc Y", linewidth=1.5)
    ax.plot(t, AccZ, color="#1f77b4", label="Acc Z", linewidth=1.5)
    ax.set_title(f"Inertial Window #{sample2plot} — Activity: {cls_label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Acceleration ($m/s^2$)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_class_exemplars(df_csv, classes_list=None, num_steps=200, fs=50.0):
    """
    Plots one representative 3-axis waveform per activity class side by side for visual comparison.
    """
    if classes_list is None:
        classes_list = sorted(df_csv["class"].unique().tolist())

    n_cls = len(classes_list)
    t = np.linspace(0, num_steps / fs, num_steps)
    fig, axes = plt.subplots(1, n_cls, figsize=(4.5 * n_cls, 3.8), sharey=True)
    if n_cls == 1:
        axes = [axes]

    for ax, cls_name in zip(axes, classes_list):
        sub = df_csv[df_csv["class"] == cls_name]
        if sub.empty:
            continue
        row = sub.iloc[0]
        AccX = row.loc["param1":f"param{num_steps}"].values.astype(float)
        AccY = row.loc[f"param{num_steps + 1}":f"param{2 * num_steps}"].values.astype(float)
        AccZ = row.loc[f"param{2 * num_steps + 1}":f"param{3 * num_steps}"].values.astype(float)

        ax.plot(t, AccX, color="#d62728", label="X", alpha=0.9)
        ax.plot(t, AccY, color="#2ca02c", label="Y", alpha=0.9)
        ax.plot(t, AccZ, color="#1f77b4", label="Z", alpha=0.9)
        ax.set_title(f"Class: {cls_name}", fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Acceleration ($m/s^2$)")
    axes[-1].legend(loc="upper right")
    plt.suptitle("3-Axis Accelerometer Signatures Across Activities", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()


def PlotPCA(pca, Acc_array, y, title="", ax=None):
    """
    Projects Acc_array onto its first 3 Principal Components and plots a 3D scatter chart
    generalized for any set of activity classes.
    """
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    principalComponents = pca.transform(Acc_array)
    classes = sorted(pd.Series(y).unique().tolist())
    palette = sns.color_palette("tab10", max(len(classes), 3))

    for cls_name, color in zip(classes, palette):
        mask = (y == cls_name).values if hasattr(y, "values") else (y == cls_name)
        ax.scatter(
            principalComponents[mask, 0],
            principalComponents[mask, 1],
            principalComponents[mask, 2],
            label=str(cls_name),
            color=color,
            s=35,
            alpha=0.8,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="best")
    if standalone:
        plt.tight_layout()
        plt.show()


def CreatePCA(train_df, test_df):
    """
    Fits 3-component PCA on the full 3-axis training signals and plots Train vs Test 3D projections.
    """
    _, _, _, AccAll_train, y_train = GetAccelerationAxis(train_df)
    _, _, _, AccAll_test, y_test = GetAccelerationAxis(test_df)

    pca_all = decomposition.PCA(n_components=3)
    pca_all.fit(AccAll_train)
    var_exp = np.sum(pca_all.explained_variance_ratio_) * 100.0

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    PlotPCA(pca_all, AccAll_train, y_train, title=f"3D PCA — TRAIN SET ({var_exp:.1f}% var)", ax=ax1)
    PlotPCA(pca_all, AccAll_test, y_test, title=f"3D PCA — TEST SET ({var_exp:.1f}% var)", ax=ax2)
    plt.tight_layout()
    plt.show()


def plot_spectral_and_cepstral_summary(df_csv, classes_list=None, num_steps=200, fs=50.0, n_mfcc=8):
    """
    Visualizes the average FFT Power Spectrum and Inertial MFCC profile per activity class.
    """
    if classes_list is None:
        classes_list = sorted(df_csv["class"].unique().tolist())

    freqs = rfftfreq(num_steps, d=1.0 / fs)[1:]
    fig, (ax_fft, ax_mfcc) = plt.subplots(1, 2, figsize=(14, 4.5))

    mfcc_means = []
    for cls_name in classes_list:
        sub = df_csv[df_csv["class"] == cls_name]
        raw = sub.iloc[:, : 3 * num_steps].values.astype(np.float32)
        smv = np.sqrt(raw[:, :num_steps] ** 2 + raw[:, num_steps:2 * num_steps] ** 2 + raw[:, 2 * num_steps:] ** 2)
        smv_centered = smv - np.mean(smv, axis=1, keepdims=True)
        psd = (np.abs(rfft(smv_centered, n=num_steps, axis=1))[:, 1:] ** 2) / num_steps
        ax_fft.plot(freqs, np.mean(psd, axis=0), label=str(cls_name), linewidth=1.8)

        imfcc = extract_imfcc_features(smv, fs=fs, n_mfcc=n_mfcc)
        mfcc_means.append(np.mean(imfcc, axis=0))

    ax_fft.set_title("Mean FFT Power Spectral Density (SMV)", fontweight="bold")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Power")
    ax_fft.legend()
    ax_fft.grid(True, alpha=0.3)

    mfcc_df = pd.DataFrame(
        mfcc_means,
        index=classes_list,
        columns=[f"IMFCC-{i + 1}" for i in range(n_mfcc)],
    )
    sns.heatmap(mfcc_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_mfcc)
    ax_mfcc.set_title("Mean Inertial Cepstral Coefficients (IMFCC) per Class", fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_decision_tree_diagram(clf, feature_names, class_names, max_depth=3):
    """Plots the top levels of a trained Scikit-Learn DecisionTreeClassifier."""
    plt.figure(figsize=(16, 7))
    plot_tree(
        clf,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title(f"Decision Tree Structure (Top {max_depth} Levels)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_feature_importances(clf, feature_names, top_n=15, title="Top Engineered Features by Importance"):
    """Plots the top_n most informative features from a tree or forest classifier."""
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices][::-1]
    top_scores = importances[indices][::-1]

    plt.figure(figsize=(9, 4.8))
    plt.barh(top_features, top_scores, color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.xlabel("Relative Importance (Gini / Entropy Gain)")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title="Sequential Model Learning Curves"):
    """Plots training and validation accuracy & loss curves from a Keras History object."""
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax1.plot(epochs_range, hist["accuracy"], label="Train Accuracy", linewidth=2)
    if "val_accuracy" in hist:
        ax1.plot(epochs_range, hist["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
    ax1.set_title(f"{title} — Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, hist["loss"], label="Train Loss", linewidth=2, color="#d62728")
    if "val_loss" in hist:
        ax2.plot(epochs_range, hist["val_loss"], label="Validation Loss", linewidth=2, linestyle="--", color="#ff7f0e")
    ax2.set_title(f"{title} — Cross-Entropy Loss", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plots a formatted Seaborn heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Activity")
    plt.ylabel("Actual Activity")
    plt.tight_layout()
    plt.show()


def plot_systems_comparison(model_accuracies, num_samples_test):
    """
    Compares multiple trained systems using 95% confidence intervals and plots
    both a bar chart and a confidence interval error-bar chart.
    :param model_accuracies: dict mapping model label -> test accuracy in % (0..100)
    :param num_samples_test: number of samples in the test set
    """
    labels = list(model_accuracies.keys())
    means = [float(model_accuracies[k]) for k in labels]
    cis = [CalculateCI(num_samples_test, acc) for acc in means]

    print(f"=== Statistical Systems Comparison (Test Set N = {num_samples_test}, 95% CI) ===")
    for label, acc, ci in zip(labels, means, cis):
        low = max(0.0, acc - ci)
        high = min(100.0, acc + ci)
        print(f"  {label:<32}: {acc:6.2f}%  ± {ci:5.2f}%  -> 95% CI: [{low:6.2f}%, {high:6.2f}%]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    colors = sns.color_palette("Set2", len(labels))
    x_pos = np.arange(len(labels))

    bars = ax1.bar(x_pos, means, yerr=cis, capsize=6, color=colors, edgecolor="black", alpha=0.9)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Accuracy Comparison with 95% Confidence Intervals", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, min(acc + 3.0, 103.0), f"{acc:.1f}%", ha="center", fontweight="bold")

    ax2.errorbar(x_pos, means, yerr=cis, fmt="o", markersize=8, capsize=8, linewidth=2, color="#1f77b4")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_xlim(-0.5, len(labels) - 0.5)
    ax2.set_ylim(max(0.0, min(means) - max(cis) - 10.0), 105.0)
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("95% Confidence Interval Overlap Analysis", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
