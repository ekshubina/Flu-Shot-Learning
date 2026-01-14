# KNN Hyperparameter Optimization Results

## Objective
Test different `n_neighbors` values for KNN imputation to find the optimal setting for ordinal feature imputation.

## Methodology
Ran full pipeline with 5-fold stratified cross-validation for each n_neighbors value:
- **Values tested**: 3, 5, 7, 10, 15
- **Baseline**: Mean/Mode imputation (AUROC 0.8441)
- **Config**: Type-based imputation with KNN for ordinals, Mode for nominal
- **Seed**: 42 (reproducibility)

## Results

| n_neighbors | Mean AUROC | H1N1 AUROC | Seasonal AUROC | vs Baseline | Delta % |
|-------------|-----------|-----------|----------------|------------|---------|
| 3           | 0.8414    | 0.8323    | 0.8505         | -0.0027    | -0.32%  |
| 5           | 0.8414    | 0.8324    | 0.8505         | -0.0027    | -0.32%  |
| **7**       | 0.8420    | 0.8330    | 0.8510         | -0.0021    | -0.25%  |
| **10** ⭐   | 0.8430    | 0.8343    | 0.8517         | -0.0011    | -0.13%  |
| 15          | 0.8426    | 0.8340    | 0.8512         | -0.0015    | -0.18%  |

## Key Findings

### Best Result: n_neighbors = 10
- **Mean AUROC**: 0.8430 (closest to baseline)
- **vs Baseline**: -0.13% (much better than n_neighbors=5 at -0.32%)
- **Improvement**: +0.16% over n_neighbors=5
- **Still below baseline** by -0.11 points (0.8441 - 0.8430)

### Pattern Observed
```
n_neighbors=3,5     → 0.8414 (worst)
n_neighbors=7       → 0.8420 (+0.06% improvement)
n_neighbors=10      → 0.8430 (+0.16% improvement) ← OPTIMAL
n_neighbors=15      → 0.8426 (-0.04% vs 10, slight regression)
```

**Interpretation**: Performance improves from n=3 to n=10, then plateaus/slightly decreases. The sweet spot is **n_neighbors=10**, balancing:
- Enough neighbors to smooth noise
- Not so many that local information is lost

## Conclusion

### Why KNN Still Underperforms (by 0.13%)

1. **Low Missingness (1-2%)**: Your data has very few missing values; simple methods already work well
2. **MCAR Pattern**: Missing values are completely random (not related to other features)
   - KNN advantage: Handling MAR (Missing At Random), where missingness depends on other features
   - Your data: MCAR (Missing Completely At Random)
   - Theory: For MCAR, mean is the statistically optimal imputation
3. **Curse of Dimensionality**: With 35 features, finding "similar" neighbors is harder
   - KNN relies on distance metrics in high-dimensional space
   - High dimensions → distances become less meaningful → neighbor selection less effective

### Recommendation

**Stick with baseline (Mean/Mode imputation)** because:
- ✅ Performs better by +0.13% (0.8441 vs 0.8430)
- ✅ Much simpler and faster
- ✅ Theoretically justified for MCAR data with low missingness
- ✅ More interpretable and reproducible

### If You Want to Explore Further

Only pursue **MICE** if you:
1. Suspect data has **MAR patterns** (check correlation between missingness and other features)
2. Have time for implementation (more complex than KNN)
3. Are willing to accept potential trade-offs for theoretical sophistication

For this dataset, **simple is better**.

## Configuration Update

Updated [config_type_based_knn.yaml](config_type_based_knn.yaml) to use optimized `n_neighbors=10`.

Command to reproduce best KNN result:
```bash
python main.py --config examples/config_type_based_knn.yaml --seed 42
```

## Script

The optimization script [test_knn_neighbors.py](../test_knn_neighbors.py) is saved for future reference and can test any n_neighbors values by editing the list.
