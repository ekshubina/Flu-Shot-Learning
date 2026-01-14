#!/usr/bin/env python3
"""
Test different n_neighbors values for KNN imputation optimization.

Runs the full pipeline with different n_neighbors settings and compares AUROC results.
"""

import subprocess
import yaml
import re
import pandas as pd
from pathlib import Path
import sys

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config_path, config_dict):
    """Save YAML configuration."""
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

def run_pipeline(config_path, seed=42):
    """Run pipeline with given config and extract results."""
    try:
        result = subprocess.run(
            ['python', 'main.py', '--config', config_path, '--seed', str(seed)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        output = result.stdout + result.stderr
        
        # Extract AUROC metrics
        auroc_mean = None
        h1n1_auroc = None
        seasonal_auroc = None
        
        # Look for "Mean AUROC: 0.8414 ± 0.0045"
        match = re.search(r'Mean AUROC:\s+([\d.]+)\s*±\s*([\d.]+)', output)
        if match:
            auroc_mean = float(match.group(1))
        
        # Look for "H1N1 AUROC: 0.8324 ± 0.0039"
        match = re.search(r'H1N1 AUROC:\s+([\d.]+)\s*±\s*([\d.]+)', output)
        if match:
            h1n1_auroc = float(match.group(1))
        
        # Look for "Seasonal AUROC: 0.8505 ± 0.0057"
        match = re.search(r'Seasonal AUROC:\s+([\d.]+)\s*±\s*([\d.]+)', output)
        if match:
            seasonal_auroc = float(match.group(1))
        
        success = auroc_mean is not None
        return {
            'success': success,
            'auroc_mean': auroc_mean,
            'h1n1_auroc': h1n1_auroc,
            'seasonal_auroc': seasonal_auroc,
            'output': output[-2000:] if not success else ""  # Last 2000 chars if failed
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'auroc_mean': None,
            'h1n1_auroc': None,
            'seasonal_auroc': None,
            'output': "TIMEOUT: Pipeline took too long"
        }
    except Exception as e:
        return {
            'success': False,
            'auroc_mean': None,
            'h1n1_auroc': None,
            'seasonal_auroc': None,
            'output': f"ERROR: {str(e)}"
        }

def main():
    config_path = Path('examples/config_type_based_knn.yaml')
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Test values
    n_neighbors_values = [3, 5, 7, 10, 15]
    results = []
    
    print("=" * 80)
    print("KNN HYPERPARAMETER OPTIMIZATION: Testing n_neighbors values")
    print("=" * 80)
    print()
    
    for n_neighbors in n_neighbors_values:
        print(f"Testing n_neighbors = {n_neighbors}...")
        
        # Load config
        config = load_config(config_path)
        
        # Modify n_neighbors
        config['imputation']['ordinal_params']['n_neighbors'] = n_neighbors
        
        # Save modified config
        temp_config_path = Path('examples/config_type_based_knn_temp.yaml')
        save_config(temp_config_path, config)
        
        # Run pipeline
        result = run_pipeline(str(temp_config_path), seed=42)
        
        results.append({
            'n_neighbors': n_neighbors,
            'auroc_mean': result['auroc_mean'],
            'h1n1_auroc': result['h1n1_auroc'],
            'seasonal_auroc': result['seasonal_auroc'],
            'success': result['success']
        })
        
        if result['success']:
            print(f"  ✓ Mean AUROC: {result['auroc_mean']:.4f}")
            print(f"    - H1N1: {result['h1n1_auroc']:.4f}")
            print(f"    - Seasonal: {result['seasonal_auroc']:.4f}")
        else:
            print(f"  ✗ FAILED")
            if result['output']:
                print(f"    {result['output'][:200]}")
        print()
    
    # Clean up temp config
    temp_config_path.unlink(missing_ok=True)
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    
    # Display results
    print("\nAll Results:")
    print("-" * 80)
    for _, row in df.iterrows():
        if row['success']:
            print(f"n_neighbors={row['n_neighbors']:2d}  →  "
                  f"Mean AUROC: {row['auroc_mean']:.4f}  |  "
                  f"H1N1: {row['h1n1_auroc']:.4f}  |  "
                  f"Seasonal: {row['seasonal_auroc']:.4f}")
        else:
            print(f"n_neighbors={row['n_neighbors']:2d}  →  FAILED")
    
    # Find best
    successful = df[df['success'] == True]
    if not successful.empty:
        best_idx = successful['auroc_mean'].idxmax()
        best_row = df.loc[best_idx]
        baseline = 0.8441
        
        print("\n" + "-" * 80)
        print(f"BEST RESULT: n_neighbors = {int(best_row['n_neighbors'])}")
        print(f"  Mean AUROC: {best_row['auroc_mean']:.4f}")
        print(f"  vs Baseline (0.8441): {best_row['auroc_mean'] - baseline:+.4f} ({(best_row['auroc_mean'] - baseline)/baseline*100:+.2f}%)")
        
        if best_row['auroc_mean'] > baseline:
            print(f"  ✓ IMPROVEMENT FOUND!")
        else:
            print(f"  ✗ Baseline still better")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
