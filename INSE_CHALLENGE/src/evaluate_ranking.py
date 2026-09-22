#!/usr/bin/env python3
"""
evaluate_ranking.py
-------------------
Automatic grading and ranking utility for the tennis HAR student challenge.
Evaluates student submission predictions against ground-truth labels.
Supports single submission evaluation and multi-submission directory leaderboard ranking.

Author: Antigravity (Assistant)
"""

import os
import sys
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

VALID_CLASSES = ['BackhandRH', 'BackhandTwoH', 'Drive', 'Lob', 'Serve', 'Waiting']


def evaluate_single_submission(sub_file: str, ground_truth_df: pd.DataFrame, student_id: str = None) -> Dict[str, Any]:
    """
    Evaluates a single student submission file against ground truth.
    """
    if student_id is None:
        student_id = os.path.splitext(os.path.basename(sub_file))[0]

    # Load submission
    if not os.path.exists(sub_file):
        raise FileNotFoundError(f"Submission file not found: {sub_file}")

    df_sub = pd.read_csv(sub_file)

    # Check required columns
    expected_cols = {'sample_id', 'predicted_label'}
    if not expected_cols.issubset(df_sub.columns):
        raise ValueError(f"Invalid submission format in {sub_file}. Required columns: {expected_cols}, got: {list(df_sub.columns)}")

    # Check row count
    if len(df_sub) != len(ground_truth_df):
        raise ValueError(f"Row count mismatch in {sub_file}: expected {len(ground_truth_df)}, got {len(df_sub)}")

    # Join on sample_id
    merged = pd.merge(ground_truth_df, df_sub, on='sample_id', how='left')
    if merged['predicted_label'].isna().any():
        missing_count = merged['predicted_label'].isna().sum()
        raise ValueError(f"Submission {sub_file} has {missing_count} unmatched or NaN predictions.")

    y_true = merged['label'].values
    y_pred = merged['predicted_label'].values

    # Check valid labels
    invalid_labels = set(y_pred) - set(VALID_CLASSES)
    if invalid_labels:
        raise ValueError(f"Submission contains invalid gesture labels: {invalid_labels}. Valid: {VALID_CLASSES}")

    # Compute Global Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=VALID_CLASSES)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=VALID_CLASSES)

    # Hardware breakdown
    device_metrics = {}
    for dev in ['Commercial', 'Custom']:
        mask = (merged['device_type'] == dev).values
        if mask.any():
            device_metrics[f"{dev}_Acc"] = accuracy_score(y_true[mask], y_pred[mask])
            device_metrics[f"{dev}_F1_Macro"] = f1_score(y_true[mask], y_pred[mask], average='macro', labels=VALID_CLASSES)

    # Subject breakdown (LOSO generalization)
    subject_metrics = {}
    unique_subs = sorted(list(merged['subject_id'].unique()), key=lambda x: int(x[1:]))
    for sub in unique_subs:
        mask = (merged['subject_id'] == sub).values
        if mask.any():
            subject_metrics[f"{sub}_Acc"] = accuracy_score(y_true[mask], y_pred[mask])

    return {
        'student_id': student_id,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'device_metrics': device_metrics,
        'subject_metrics': subject_metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'merged_df': merged
    }


def print_detailed_report(result: Dict[str, Any]):
    """Prints a clean CLI report for a single evaluation."""
    student_id = result['student_id']
    acc = result['accuracy'] * 100
    bal_acc = result['balanced_accuracy'] * 100
    f1_m = result['f1_macro'] * 100
    f1_w = result['f1_weighted'] * 100

    print("=" * 80)
    print(f"  EVALUATION REPORT: {student_id}")
    print("=" * 80)
    print(f"  Global Accuracy:          {acc:6.2f}%")
    print(f"  Balanced Accuracy:        {bal_acc:6.2f}%")
    print(f"  Macro F1-Score:           {f1_m:6.2f}%")
    print(f"  Weighted F1-Score:        {f1_w:6.2f}%")
    print("-" * 80)
    print("  Cross-Hardware Performance:")
    for k, v in result['device_metrics'].items():
        print(f"    - {k:20s}: {v*100:6.2f}%")
    print("-" * 80)
    print("  Per-Subject Performance (LOSO Generalization):")
    for k, v in result['subject_metrics'].items():
        print(f"    - Subject {k:12s}: {v*100:6.2f}%")
    print("-" * 80)
    print("  Classification Report (Per-Gesture):")
    report = classification_report(result['y_true'], result['y_pred'], labels=VALID_CLASSES, digits=4)
    print(report)
    print("-" * 80)
    print("  Confusion Matrix:")
    cm = confusion_matrix(result['y_true'], result['y_pred'], labels=VALID_CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in VALID_CLASSES], columns=[f"Pred_{c}" for c in VALID_CLASSES])
    print(cm_df.to_string())
    print("=" * 80)


def build_leaderboard(submissions_dir: str, ground_truth_df: pd.DataFrame, output_csv: str = None) -> pd.DataFrame:
    """
    Evaluates all CSV submissions found in submissions_dir and ranks them.
    """
    files = [f for f in os.listdir(submissions_dir) if f.endswith('.csv') and f != "sample_submission.csv" and "leaderboard" not in f.lower()]
    if not files:
        print(f"No submission CSV files found in {submissions_dir}")
        return pd.DataFrame()

    results = []
    for f in sorted(files):
        fpath = os.path.join(submissions_dir, f)
        try:
            res = evaluate_single_submission(fpath, ground_truth_df)
            entry = {
                'Student/Model': res['student_id'],
                'Macro_F1 (%)': round(res['f1_macro'] * 100, 2),
                'Accuracy (%)': round(res['accuracy'] * 100, 2),
                'Balanced_Acc (%)': round(res['balanced_accuracy'] * 100, 2),
                'Commercial_F1 (%)': round(res['device_metrics'].get('Commercial_F1_Macro', 0) * 100, 2),
                'Custom_F1 (%)': round(res['device_metrics'].get('Custom_F1_Macro', 0) * 100, 2),
            }
            # Append per-subject accuracy
            for sub_k, sub_v in res['subject_metrics'].items():
                entry[f"{sub_k} (%)"] = round(sub_v * 100, 2)
            results.append(entry)
        except Exception as e:
            print(f"[Error] Failed to evaluate {f}: {e}")

    df_leaderboard = pd.DataFrame(results)
    if not df_leaderboard.empty:
        # Sort by Macro_F1 descending, then Accuracy descending
        df_leaderboard.sort_values(by=['Macro_F1 (%)', 'Accuracy (%)'], ascending=[False, False], inplace=True)
        df_leaderboard.reset_index(drop=True, inplace=True)
        df_leaderboard.index += 1  # 1-based ranking rank
        df_leaderboard.index.name = 'Rank'

    if output_csv and not df_leaderboard.empty:
        df_leaderboard.to_csv(output_csv)
        print(f"[Leaderboard Saved] Exported ranking table to {output_csv}")

    return df_leaderboard


def main():
    parser = argparse.ArgumentParser(description="Evaluate student challenge submissions.")
    parser.add_argument("--ground_truth", type=str, default="./challenge_data/test_labeled.csv",
                        help="Path to ground truth test_labeled.csv (or .parquet)")
    parser.add_argument("--submission", type=str, default=None,
                        help="Path to single student submission CSV file")
    parser.add_argument("--submissions_dir", type=str, default=None,
                        help="Directory containing multiple student submission CSV files")
    parser.add_argument("--output_leaderboard", type=str, default=None,
                        help="Path to export leaderboard CSV")

    args = parser.parse_args()

    # Load ground truth
    if not os.path.exists(args.ground_truth):
        # Check parquet fallback
        alt_gt = os.path.splitext(args.ground_truth)[0] + ".parquet"
        if os.path.exists(alt_gt):
            args.ground_truth = alt_gt
        else:
            raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")

    if args.ground_truth.endswith('.parquet'):
        try:
            gt_df = pd.read_parquet(args.ground_truth)
        except (ImportError, ValueError):
            alt_csv = os.path.splitext(args.ground_truth)[0] + '.csv'
            if os.path.exists(alt_csv):
                print(f'[Notice] Parquet engine not found. Automatically loaded CSV fallback: {alt_csv}')
                gt_df = pd.read_csv(alt_csv)
            else:
                raise
    else:
        gt_df = pd.read_csv(args.ground_truth)

    if args.submission:
        res = evaluate_single_submission(args.submission, gt_df)
        print_detailed_report(res)

    if args.submissions_dir:
        print(f"\n[Compiling Leaderboard from: {args.submissions_dir}]")
        lb = build_leaderboard(args.submissions_dir, gt_df, output_csv=args.output_leaderboard)
        print("\n" + lb.to_string())


if __name__ == "__main__":
    main()
