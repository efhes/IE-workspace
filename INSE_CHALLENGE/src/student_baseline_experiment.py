#!/usr/bin/env python3
"""
student_baseline_experiment.py
------------------------------
Sample student experiment script using Traditional Machine Learning Baselines
with Frontend Functional / Statistical Feature Engineering.

Frontend Scheme:
  - Input: Raw sensor tensor windows of shape (N, 200, 12).
  - Feature Extraction: Computes 6 statistical descriptors along the temporal window axis:
    (mean, std, var, min, max, median) across the 12 channels.
  - Dimension Reduction: (N, 200, 12) -> (N, 72).

Models Evaluated:
  1. ZeroR (Dummy Most-Frequent Classifier)
  2. Naive Bayes (GaussianNB)
  3. Decision Tree (CART)
  4. Random Forest (Ensemble)

Workflow:
  1. Loads Train and Blind Test datasets via challenge_utils.
  2. Extracts the 72-dimensional functional vectors.
  3. Evaluates internal LOSO Cross-Validation on the 7 training subjects.
  4. Trains each model on the full training set.
  5. Generates official competition submission CSV files for all models.
  6. Optionally evaluates submissions against ground truth using evaluate_ranking.py.

Author: Antigravity (Assistant)
"""

import os
import sys
import argparse
import time
import warnings
from typing import Dict, Any
import numpy as np
import pandas as pd

# Filter runtime optimization warnings for clean CLI output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Challenge helper library
from challenge_utils import load_train, load_test, compute_functionals, create_submission, VALID_CLASSES


def get_baseline_classifiers() -> Dict[str, Any]:
    """
    Instantiates the 6 traditional baseline classifiers.
    StandardScaler is embedded via Pipeline for scale-sensitive models (LR, SVM).
    """
    models = {
        "ZeroR": DummyClassifier(strategy='most_frequent'),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    return models


def run_internal_loso_validation(X_feats: np.ndarray, y: np.ndarray, subjects: np.ndarray, models: Dict[str, Any]):
    """
    Performs internal Leave-One-Subject-Out (LOSO) Cross-Validation across training subjects.
    Provides students with realistic out-of-fold generalization estimates.
    """
    unique_subs = sorted(list(np.unique(subjects)), key=lambda x: int(x[1:]))
    print("\n" + "=" * 80)
    print(f"  INTERNAL LOSO CROSS-VALIDATION ON TRAINING SET ({len(unique_subs)} subjects: {unique_subs})")
    print("=" * 80)

    cv_summary = {}

    for name, model in models.items():
        start_time = time.time()
        fold_accs = []
        fold_f1s = []
        y_true_all = []
        y_pred_all = []

        print(f"\nEvaluating: {name:<20s} ...", end="", flush=True)

        for val_sub in unique_subs:
            train_mask = (subjects != val_sub)
            val_mask = (subjects == val_sub)

            X_tr, y_tr = X_feats[train_mask], y[train_mask]
            X_val, y_val = X_feats[val_mask], y[val_mask]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average='macro', labels=VALID_CLASSES)

            fold_accs.append(acc)
            fold_f1s.append(f1)
            y_true_all.extend(y_val)
            y_pred_all.extend(preds)

        elapsed = time.time() - start_time
        mean_acc = np.mean(fold_accs) * 100
        std_acc = np.std(fold_accs) * 100
        mean_f1 = np.mean(fold_f1s) * 100
        std_f1 = np.std(fold_f1s) * 100
        ci95 = 1.96 * (std_acc / np.sqrt(len(fold_accs)))

        cv_summary[name] = {
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'ci95': ci95,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'time_sec': elapsed
        }

        print(f" Done! Mean Acc: {mean_acc:6.2f}% ± {ci95:4.2f}% (95% CI) | Macro F1: {mean_f1:6.2f}% [{elapsed:5.1f}s]")

    print("\n" + "=" * 80)
    print("  INTERNAL VALIDATION SUMMARY TABLE (Ranked by Macro F1)")
    print("=" * 80)
    ranked = sorted(cv_summary.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
    print(f"  {'Rank':<5s} {'Classifier':<22s} {'Macro F1 (%)':<15s} {'Accuracy (%)':<20s} {'Training Time':<12s}")
    print("  " + "-" * 76)
    for idx, (m_name, res) in enumerate(ranked, 1):
        acc_str = f"{res['mean_acc']:5.2f} ± {res['ci95']:4.2f}"
        f1_str = f"{res['mean_f1']:5.2f} ± {res['std_f1']:4.2f}"
        time_str = f"{res['time_sec']:5.1f} s"
        print(f"  {idx:<5d} {m_name:<22s} {f1_str:<15s} {acc_str:<20s} {time_str:<12s}")
    print("=" * 80)


def generate_challenge_submissions(X_train_feats: np.ndarray, y_train: np.ndarray,
                                    X_test_feats: np.ndarray, test_meta: pd.DataFrame,
                                    models: Dict[str, Any], submissions_dir: str):
    """
    Fits each model on the complete training set (10,080 samples) and predicts on the blind test set (7,200 samples).
    Saves individual submission CSV files matching the challenge requirements.
    """
    os.makedirs(submissions_dir, exist_ok=True)
    print(f"\n[Generating Challenge Submissions -> {submissions_dir}]")
    test_sample_ids = test_meta['sample_id'].values

    for name, model in models.items():
        print(f"  Fitting {name:<20s} on full training set ...", end="", flush=True)
        start = time.time()
        model.fit(X_train_feats, y_train)
        y_test_pred = model.predict(X_test_feats)
        elapsed = time.time() - start

        safe_name = name.lower().replace(" ", "_")
        sub_path = os.path.join(submissions_dir, f"submission_{safe_name}.csv")
        create_submission(test_sample_ids, y_test_pred, output_file=sub_path)
        print(f" [{elapsed:4.1f}s]")


def main():
    parser = argparse.ArgumentParser(description="Student Experiment: Traditional Baseline Classifiers on Functional HAR Features.")
    parser.add_argument("--data_dir", type=str, default="./challenge_data",
                        help="Path to directory containing challenge files")
    parser.add_argument("--submissions_dir", type=str, default="./student_submissions",
                        help="Directory where predicted submission CSV files will be saved")
    parser.add_argument("--skip_loso", action="store_true",
                        help="Skip internal LOSO CV and proceed directly to full training & test prediction")
    parser.add_argument("--evaluate", action="store_true", default=True,
                        help="Automatically evaluate submissions if ground-truth test_labeled.csv is accessible")

    args = parser.parse_args()

    print("=" * 80)
    print("  TENNIS HAR CHALLENGE: TRADITIONAL BASELINE CLASSIFIERS EXPERIMENT")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[Step 1] Loading Challenge Datasets...")
    X_train_raw, y_train, meta_train = load_train(args.data_dir, prefer_format='parquet')
    X_test_raw, meta_test = load_test(args.data_dir, labeled=False, prefer_format='parquet')

    print(f"  Raw Train Tensor: {X_train_raw.shape} ({X_train_raw.nbytes / (1024*1024):.1f} MB)")
    print(f"  Raw Test Tensor:  {X_test_raw.shape} ({X_test_raw.nbytes / (1024*1024):.1f} MB)")
    print(f"  Activity Classes: {VALID_CLASSES}")

    # 2. Frontend Feature Engineering (Functionals)
    print("\n[Step 2] Frontend Feature Extraction: 6 Functionals across 12 Channels...")
    t0 = time.time()
    X_train_feats = compute_functionals(X_train_raw)
    X_test_feats = compute_functionals(X_test_raw)
    t_feat = time.time() - t0

    print(f"  Extracted Train Features: {X_train_feats.shape} [Reduced from 200 time steps x 12 channels -> 72 features]")
    print(f"  Extracted Test Features:  {X_test_feats.shape}")
    print(f"  Computation Time:         {t_feat:.2f} seconds")

    # 3. Instantiate Models
    models = get_baseline_classifiers()

    # 4. Internal LOSO Cross-Validation
    if not args.skip_loso:
        run_internal_loso_validation(X_train_feats, y_train, meta_train['subject_id'].values, models)

    # 5. Generate Test Submissions
    generate_challenge_submissions(X_train_feats, y_train, X_test_feats, meta_test, models, args.submissions_dir)

    # 6. Optional Instructor Ground-Truth Evaluation
    gt_file = os.path.join(args.data_dir, "test_labeled.parquet")
    if not os.path.exists(gt_file):
        gt_file = os.path.join(args.data_dir, "test_labeled.csv")

    if args.evaluate and os.path.exists(gt_file):
        print("\n" + "=" * 80)
        print("  OFFICIAL CHALLENGE LEADERBOARD (Evaluated on Hidden Test Set)")
        print("=" * 80)
        from evaluate_ranking import build_leaderboard
        gt_df = pd.read_parquet(gt_file) if gt_file.endswith('.parquet') else pd.read_csv(gt_file)
        leaderboard = build_leaderboard(args.submissions_dir, gt_df,
                                        output_csv=os.path.join(args.submissions_dir, "leaderboard.csv"))
        print("\n" + leaderboard.to_string())

    print("\n[Done] Experiment finished successfully!")


if __name__ == "__main__":
    main()
