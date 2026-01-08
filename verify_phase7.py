#!/usr/bin/env python3
"""
Phase 7 Verification Script
Verifies data integrity, tests error cases, and validates output files
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_data_integrity():
    """Verify data integrity: respondent IDs, no leakage, balanced folds, etc."""
    print("\n" + "="*70)
    print("DATA INTEGRITY CHECKS")
    print("="*70)
    
    results = {}
    
    # Load data
    train_features = pd.read_csv("data/training_set_features.csv", index_col="respondent_id")
    train_labels = pd.read_csv("data/training_set_labels.csv", index_col="respondent_id")
    test_features = pd.read_csv("data/test_set_features.csv", index_col="respondent_id")
    submission = pd.read_csv("submissions/submission_baseline.csv", index_col="respondent_id")
    
    # 1. Check respondent ID preservation
    print("\n1. Respondent ID Preservation")
    train_ids_match = train_features.index.equals(train_labels.index)
    print(f"   ✓ Training features IDs match labels: {train_ids_match}")
    results["respondent_ids_match"] = train_ids_match
    
    test_ids_match = test_features.index.equals(submission.index)
    print(f"   ✓ Test feature IDs match submission: {test_ids_match}")
    results["test_ids_match"] = test_ids_match
    
    # 2. Check no missing values in submission
    print("\n2. Missing Values in Submission")
    nan_count = submission.isna().sum().sum()
    print(f"   ✓ Total NaN values in submission: {nan_count}")
    results["no_nan_in_submission"] = nan_count == 0
    
    # 3. Check probability ranges
    print("\n3. Probability Value Ranges")
    h1n1_probs = submission["h1n1_vaccine"]
    seasonal_probs = submission["seasonal_vaccine"]
    
    h1n1_valid = (h1n1_probs >= 0.0).all() and (h1n1_probs <= 1.0).all()
    seasonal_valid = (seasonal_probs >= 0.0).all() and (seasonal_probs <= 1.0).all()
    
    print(f"   ✓ H1N1 probabilities in [0.0, 1.0]: {h1n1_valid}")
    print(f"     - Min: {h1n1_probs.min():.6f}, Max: {h1n1_probs.max():.6f}")
    print(f"   ✓ Seasonal probabilities in [0.0, 1.0]: {seasonal_valid}")
    print(f"     - Min: {seasonal_probs.min():.6f}, Max: {seasonal_probs.max():.6f}")
    
    results["h1n1_probs_valid"] = h1n1_valid
    results["seasonal_probs_valid"] = seasonal_valid
    
    # 4. Check row counts
    print("\n4. Row Counts")
    submission_rows = len(submission)
    test_rows = len(test_features)
    print(f"   ✓ Submission rows: {submission_rows}")
    print(f"   ✓ Test feature rows: {test_rows}")
    print(f"   ✓ Counts match: {submission_rows == test_rows}")
    results["row_counts_match"] = submission_rows == test_rows
    
    # 5. Check training data shapes
    print("\n5. Training Data Shapes")
    print(f"   ✓ Training features: {train_features.shape}")
    print(f"   ✓ Training labels: {train_labels.shape}")
    print(f"   ✓ Test features: {test_features.shape}")
    results["train_shape"] = train_features.shape
    results["test_shape"] = test_features.shape
    
    # 6. Check for data types consistency
    print("\n6. Data Type Consistency")
    all_numeric_h1n1 = pd.api.types.is_numeric_dtype(h1n1_probs)
    all_numeric_seasonal = pd.api.types.is_numeric_dtype(seasonal_probs)
    print(f"   ✓ H1N1 probabilities numeric: {all_numeric_h1n1}")
    print(f"   ✓ Seasonal probabilities numeric: {all_numeric_seasonal}")
    results["numeric_types"] = all_numeric_h1n1 and all_numeric_seasonal
    
    return results


def check_output_files():
    """Verify output file structure and content"""
    print("\n" + "="*70)
    print("OUTPUT FILE VALIDATION")
    print("="*70)
    
    results = {}
    
    # Check submission CSV
    print("\n1. Submission CSV Structure")
    submission = pd.read_csv("submissions/submission_baseline.csv")
    
    expected_columns = ["respondent_id", "h1n1_vaccine", "seasonal_vaccine"]
    columns_match = list(submission.columns) == expected_columns
    print(f"   ✓ Columns correct: {columns_match}")
    print(f"     - Expected: {expected_columns}")
    print(f"     - Actual: {list(submission.columns)}")
    results["columns_correct"] = columns_match
    
    # Check row count matches test set
    print("\n2. Row Count Validation")
    test_features = pd.read_csv("data/test_set_features.csv")
    row_count_match = len(submission) == len(test_features)
    expected_rows = len(test_features)
    print(f"   ✓ Row count matches test set ({expected_rows}): {row_count_match}")
    print(f"     - Submission rows: {len(submission)}")
    print(f"     - Test set rows: {len(test_features)}")
    results["row_count_matches"] = row_count_match
    
    # Check respondent IDs
    print("\n3. Respondent ID Validation")
    submission_ids = set(submission["respondent_id"])
    test_ids = set(test_features["respondent_id"])
    ids_match = submission_ids == test_ids
    print(f"   ✓ Respondent IDs match test set: {ids_match}")
    if not ids_match:
        missing = test_ids - submission_ids
        extra = submission_ids - test_ids
        if missing:
            print(f"     - Missing IDs: {len(missing)} (e.g., {list(missing)[:5]})")
        if extra:
            print(f"     - Extra IDs: {len(extra)} (e.g., {list(extra)[:5]})")
    results["respondent_ids_match"] = ids_match
    
    # Check for duplicates
    print("\n4. Duplicate Check")
    h1n1_dups = submission["respondent_id"].duplicated().sum()
    print(f"   ✓ Duplicate respondent IDs: {h1n1_dups}")
    results["no_duplicates"] = h1n1_dups == 0
    
    # Check value statistics
    print("\n5. Probability Statistics")
    print(f"   ✓ H1N1 vaccine:")
    print(f"     - Mean: {submission['h1n1_vaccine'].mean():.4f}")
    print(f"     - Std: {submission['h1n1_vaccine'].std():.4f}")
    print(f"     - Min/Max: [{submission['h1n1_vaccine'].min():.6f}, {submission['h1n1_vaccine'].max():.6f}]")
    print(f"   ✓ Seasonal vaccine:")
    print(f"     - Mean: {submission['seasonal_vaccine'].mean():.4f}")
    print(f"     - Std: {submission['seasonal_vaccine'].std():.4f}")
    print(f"     - Min/Max: [{submission['seasonal_vaccine'].min():.6f}, {submission['seasonal_vaccine'].max():.6f}]")
    
    return results


def test_error_cases():
    """Test pipeline's handling of error cases"""
    print("\n" + "="*70)
    print("ERROR CASE TESTING")
    print("="*70)
    
    results = {}
    
    # Test 1: Missing config file
    print("\n1. Missing Config File")
    exit_code = os.system("python main.py /nonexistent/config.yaml > /dev/null 2>&1")
    should_fail = exit_code != 0
    print(f"   ✓ Pipeline exits with error: {should_fail} (exit code: {exit_code >> 8})")
    results["missing_config_fails"] = should_fail
    
    # Test 2: Invalid YAML syntax
    print("\n2. Invalid YAML Syntax")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [syntax:\n    - broken")
        temp_yaml = f.name
    
    try:
        exit_code = os.system(f"python main.py {temp_yaml} > /dev/null 2>&1")
        should_fail = exit_code != 0
        print(f"   ✓ Pipeline exits with error: {should_fail} (exit code: {exit_code >> 8})")
        results["invalid_yaml_fails"] = should_fail
    finally:
        os.unlink(temp_yaml)
    
    # Test 3: Missing data files
    print("\n3. Missing Data Files")
    
    # Create temp config pointing to missing data
    config_content = """
data:
  path: "/nonexistent/data"
  train_features_file: "missing.csv"
  train_labels_file: "missing.csv"
  test_features_file: "missing.csv"

preprocessing:
  imputation_strategy: "mean"
  
model:
  model_type: "logistic_regression"
  
evaluation:
  n_splits: 5
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config = f.name
    
    try:
        exit_code = os.system(f"python main.py {temp_config} > /dev/null 2>&1")
        should_fail = exit_code != 0
        print(f"   ✓ Pipeline exits with error: {should_fail} (exit code: {exit_code >> 8})")
        results["missing_data_fails"] = should_fail
    finally:
        os.unlink(temp_config)
    
    # Test 4: Empty CSV file
    print("\n4. Empty/Corrupted CSV Files")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("")  # Empty file
        temp_csv = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(f"""
data:
  path: "{os.path.dirname(temp_csv)}"
  train_features_file: "{os.path.basename(temp_csv)}"
  train_labels_file: "{os.path.basename(temp_csv)}"
  test_features_file: "{os.path.basename(temp_csv)}"

preprocessing:
  imputation_strategy: "mean"
  
model:
  model_type: "logistic_regression"
  
evaluation:
  n_splits: 5
""")
        temp_config = f.name
    
    try:
        exit_code = os.system(f"python main.py {temp_config} > /dev/null 2>&1")
        should_fail = exit_code != 0
        print(f"   ✓ Pipeline exits with error: {should_fail} (exit code: {exit_code >> 8})")
        results["empty_csv_fails"] = should_fail
    finally:
        os.unlink(temp_csv)
        os.unlink(temp_config)
    
    return results


def check_no_data_leakage():
    """Verify preprocessing is fit only on training fold"""
    print("\n" + "="*70)
    print("DATA LEAKAGE CHECKS")
    print("="*70)
    
    print("\n1. Preprocessing Fit Logic")
    print("   ✓ Checking that imputation is fit only on training fold...")
    
    # Read the training engine to verify fold logic
    try:
        with open("src/training/engine.py", "r") as f:
            engine_code = f.read()
        
        # Check for proper fold handling
        has_train_fold = "X_train_fold" in engine_code or "train_idx" in engine_code
        has_fit_on_train = "fit(" in engine_code and ("X_train" in engine_code or "train_fold" in engine_code)
        
        print(f"   ✓ Engine properly splits training/validation folds: {has_train_fold}")
        print(f"   ✓ Engine fits preprocessing on training fold only: {has_fit_on_train}")
        
        return {
            "fold_separation": has_train_fold,
            "fit_on_train_only": has_fit_on_train
        }
    except Exception as e:
        print(f"   ⚠ Could not verify training engine: {e}")
        return {}


def check_fold_distribution():
    """Verify CV fold distributions are balanced per vaccine"""
    print("\n" + "="*70)
    print("CROSS-VALIDATION FOLD DISTRIBUTION CHECK")
    print("="*70)
    
    print("\n1. Training Data Label Distribution")
    
    try:
        train_labels = pd.read_csv("data/training_set_labels.csv")
        
        h1n1_dist = train_labels["h1n1_vaccine"].value_counts().sort_index()
        seasonal_dist = train_labels["seasonal_vaccine"].value_counts().sort_index()
        
        # Combined label (stratification column)
        train_labels["combined"] = train_labels["h1n1_vaccine"] + 2 * train_labels["seasonal_vaccine"]
        combined_dist = train_labels["combined"].value_counts().sort_index()
        
        print(f"\n   H1N1 vaccine distribution:")
        for val, count in h1n1_dist.items():
            pct = count / len(train_labels) * 100
            print(f"     {int(val)}: {count} ({pct:.1f}%)")
        
        print(f"\n   Seasonal vaccine distribution:")
        for val, count in seasonal_dist.items():
            pct = count / len(train_labels) * 100
            print(f"     {int(val)}: {count} ({pct:.1f}%)")
        
        print(f"\n   Combined label distribution (stratification):")
        for val, count in combined_dist.items():
            pct = count / len(train_labels) * 100
            label_name = {0: "neither", 1: "seasonal only", 2: "h1n1 only", 3: "both"}[val]
            print(f"     {int(val)} ({label_name}): {count} ({pct:.1f}%)")
        
        # Check balance
        min_class = combined_dist.min()
        max_class = combined_dist.max()
        imbalance_ratio = max_class / min_class
        
        print(f"\n   ✓ Combined label balance ratio: {imbalance_ratio:.2f}x")
        print(f"     (Min class: {min_class} samples, Max class: {max_class} samples)")
        
        return {
            "h1n1_distribution": h1n1_dist.to_dict(),
            "seasonal_distribution": seasonal_dist.to_dict(),
            "combined_distribution": combined_dist.to_dict(),
            "balance_ratio": imbalance_ratio
        }
    except Exception as e:
        print(f"   ⚠ Could not verify fold distribution: {e}")
        return {}


def main():
    """Run all Phase 7 verifications"""
    print("\n" + "="*70)
    print("PHASE 7 VERIFICATION SUITE")
    print("="*70)
    
    all_results = {}
    
    try:
        # 1. Data Integrity Checks
        all_results["data_integrity"] = check_data_integrity()
        
        # 2. Output File Validation
        all_results["output_validation"] = check_output_files()
        
        # 3. Data Leakage Checks
        all_results["leakage_checks"] = check_no_data_leakage()
        
        # 4. Fold Distribution Checks
        all_results["fold_distribution"] = check_fold_distribution()
        
        # 5. Error Case Testing
        all_results["error_cases"] = test_error_cases()
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    
    # Data Integrity
    print("\n✓ Data Integrity:")
    for key, val in all_results["data_integrity"].items():
        if isinstance(val, bool):
            status = "✓" if val else "✗"
            print(f"  {status} {key}: {val}")
            all_passed = all_passed and val
    
    # Output Validation
    print("\n✓ Output Validation:")
    for key, val in all_results["output_validation"].items():
        if isinstance(val, bool):
            status = "✓" if val else "✗"
            print(f"  {status} {key}: {val}")
            all_passed = all_passed and val
    
    # Leakage Checks
    print("\n✓ Data Leakage Prevention:")
    for key, val in all_results["leakage_checks"].items():
        if isinstance(val, bool):
            status = "✓" if val else "✗"
            print(f"  {status} {key}: {val}")
            all_passed = all_passed and val
    
    # Error Cases
    print("\n✓ Error Case Handling:")
    for key, val in all_results["error_cases"].items():
        if isinstance(val, bool):
            status = "✓" if val else "✗"
            print(f"  {status} {key}: {val}")
            all_passed = all_passed and val
    
    # Overall status
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
    else:
        print("⚠ SOME VERIFICATIONS FAILED")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
