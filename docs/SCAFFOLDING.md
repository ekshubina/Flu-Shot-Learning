# Scaffolding Guide: Extending the ML Pipeline

This guide explains how to extend and modify the ML pipeline scaffolding to fit your needs.

## Overview

The pipeline is built with a **plugin architecture** based on Abstract Base Classes (ABCs). This allows you to:

- Swap components without modifying other parts
- Add new implementations (models, imputation strategies, etc.)
- Maintain backward compatibility
- Run parallel experiments with different configurations

## Adding a New ML Model

### Step 1: Create Model File

Create `src/models/my_model.py`:

```python
from typing import Optional, Dict, Any
import numpy as np
from src.models.base import BaseModel

class MyCustomModel(BaseModel):
    """
    Custom ML model implementation.
    
    Reference: SYSTEM_DESIGN.md - Component 4: Models
    """
    
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        """Initialize model with hyperparameters."""
        self.hyperparameters = hyperparameters or {}
        self.model = None  # Actual sklearn/xgboost/custom model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MyCustomModel':
        """
        Train the model on training data.
        
        Parameters:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        
        Returns:
            self
        """
        # TODO: Implement training
        raise NotImplementedError("fit() not yet implemented")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test data.
        
        Parameters:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability predictions (n_samples, 2) with values in [0, 1]
        """
        # TODO: Implement prediction
        raise NotImplementedError("predict_proba() not yet implemented")
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        # TODO: Extract importance from trained model
        raise NotImplementedError("get_feature_importance() not yet implemented")
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return self.hyperparameters.copy()
    
    def set_params(self, **params) -> 'MyCustomModel':
        """Set model parameters."""
        self.hyperparameters.update(params)
        return self
```

### Step 2: Register in ModelFactory

Edit `src/models/factory.py`:

```python
def create_model(model_type: str, hyperparameters: Dict[str, Any]) -> BaseModel:
    """Create model instance by type."""
    
    if model_type == 'logistic_regression':
        return LogisticRegressionModel(hyperparameters)
    elif model_type == 'xgboost':
        return XGBoostModel(hyperparameters)
    elif model_type == 'my_custom_model':  # Add this
        from src.models.my_model import MyCustomModel
        return MyCustomModel(hyperparameters)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### Step 3: Update Configuration

Update your config YAML:

```yaml
model:
  model_type: my_custom_model
  hyperparameters:
    param1: value1
    param2: value2
```

### Step 4: Use in Pipeline

```python
from src.models.factory import ModelFactory

factory = ModelFactory()
model = factory.create_model('my_custom_model', config.model.hyperparameters)
```

## Adding a New Imputation Strategy

### Step 1: Create Imputation Class

Add to `src/preprocessing/imputation.py`:

```python
class MyImputation(ImputationStrategy):
    """
    Custom imputation strategy.
    
    Reference: SYSTEM_DESIGN.md - Component 2: Preprocessing
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """Initialize with parameters."""
        self.parameters = parameters or {}
        self.fit_data = None
    
    def fit(self, X: pd.DataFrame) -> 'MyImputation':
        """
        Learn imputation parameters from training data.
        
        Parameters:
            X: Training features with missing values
        
        Returns:
            self
        """
        # TODO: Learn parameters from X (e.g., compute means)
        raise NotImplementedError("fit() not yet implemented")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to data.
        
        Parameters:
            X: Features with missing values
        
        Returns:
            Features with missing values filled
        """
        # TODO: Implement imputation logic
        raise NotImplementedError("transform() not yet implemented")
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
```

### Step 2: Update Config and Usage

```yaml
imputation:
  strategy: my_imputation
  parameters:
    param1: value1
```

Then in pipeline:

```python
if config.imputation.strategy == 'my_imputation':
    from src.preprocessing.imputation import MyImputation
    imputer = MyImputation(config.imputation.parameters)
```

## Adding a New Feature Encoder

### Step 1: Create Encoder Class

Add to `src/preprocessing/encoding.py`:

```python
class MyEncoder(FeatureEncoder):
    """
    Custom feature encoding strategy.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'MyEncoder':
        """Learn encoding from training data."""
        # TODO: Implement
        raise NotImplementedError("fit() not yet implemented")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding to data."""
        # TODO: Implement
        raise NotImplementedError("transform() not yet implemented")
    
    def get_feature_names(self) -> List[str]:
        """Return names of output features."""
        # TODO: Implement
        raise NotImplementedError("get_feature_names() not yet implemented")
```

### Step 2: Update Config

```yaml
encoding:
  strategies:
    demographic:
      type: my_encoder
      parameters:
        param1: value1
```

## Adding a New Calibration Method

### Step 1: Create Calibrator

Add to `src/calibration/calibrator.py`:

```python
class MyCalibrator(CalibratorInterface):
    """
    Custom probability calibration method.
    """
    
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> 'MyCalibrator':
        """
        Learn calibration on validation predictions.
        
        Parameters:
            y_pred: Predicted probabilities (n, 2)
            y_true: True labels (n, 2)
        """
        # TODO: Learn calibration parameters
        raise NotImplementedError("fit() not yet implemented")
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply calibration to predictions.
        
        Parameters:
            y_pred: Predicted probabilities (n, 2)
        
        Returns:
            Calibrated probabilities (n, 2) in [0, 1]
        """
        # TODO: Apply calibration
        raise NotImplementedError("transform() not yet implemented")
    
    def get_calibration_error(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute calibration error."""
        # TODO: Implement
        raise NotImplementedError("get_calibration_error() not yet implemented")
```

### Step 2: Register in Factory

Update `src/calibration/calibrator.py`:

```python
def create_calibrator(method: str) -> CalibratorInterface:
    """Create calibrator by method name."""
    if method == 'none':
        return NoCalibration()
    elif method == 'my_calibrator':
        return MyCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
```

## Adding a New Evaluation Metric

### Step 1: Create Metric Function

Add to `src/utils/metrics.py`:

```python
def compute_my_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute custom metric.
    
    Parameters:
        y_true: True binary labels (n,)
        y_pred: Predicted probabilities (n,) in [0, 1]
    
    Returns:
        Metric value
    """
    # TODO: Implement
    raise NotImplementedError("compute_my_metric() not yet implemented")
```

### Step 2: Add to Evaluator

Update `src/evaluation/metrics.py`:

```python
class Evaluator:
    def get_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute all evaluation metrics."""
        # ... existing code ...
        
        diagnostics['my_metric_h1n1'] = compute_my_metric(y_true[:, 0], y_pred[:, 0])
        diagnostics['my_metric_seasonal'] = compute_my_metric(y_true[:, 1], y_pred[:, 1])
        
        return diagnostics
```

## Configuration Best Practices

### 1. Keep Base and Variant Configs

```yaml
# config_base.yaml - Common settings
data:
  train_features_path: data/training_set_features.csv
  # ...

# config_experiment1.yaml - Override specific settings
# Inherits from base
model:
  model_type: xgboost
  hyperparameters:
    max_depth: 8
```

### 2. Use Comments for Context

```yaml
# Why this value?
learning_rate: 0.05  # Careful not to overshoot, use line search

# What are the alternatives?
class_weight_strategy: balanced  # Other options: custom, none, log_balanced
```

### 3. Version Your Configs

```yaml
# At top of file
_metadata:
  version: 1.0
  created: 2026-01-07
  description: "Baseline logistic regression pipeline"
  expected_auroc: 0.80
```

## Testing New Components

### 1. Unit Test Template

Create `tests/test_my_model.py`:

```python
import numpy as np
import pytest
from src.models.my_model import MyCustomModel

def test_my_model_initialization():
    model = MyCustomModel({'param1': 10})
    assert model.hyperparameters['param1'] == 10

def test_my_model_fit_predict():
    X = np.random.randn(100, 5)
    y = np.random.binomial(1, 0.5, 100)
    
    model = MyCustomModel()
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    
    assert y_pred.shape == (100, 2)
    assert np.all((y_pred >= 0) & (y_pred <= 1))
```

### 2. Integration Test

```python
def test_model_in_pipeline():
    config = load_config('examples/config_my_model.yaml')
    results = run_pipeline(config)
    
    assert results['success']
    assert 'metrics' in results
    assert results['metrics']['auroc_mean'] > 0.7
```

## Debugging Tips

### 1. Enable Verbose Logging

```bash
python main.py --config config.yaml --verbose
```

### 2. Add Debug Output

```python
logger.debug(f"X_train shape: {X_train.shape}")
logger.debug(f"Missing values: {X_train.isna().sum().sum()}")
```

### 3. Check Data at Each Stage

```python
# In preprocessing
print(f"After imputation: {X_imputed.isna().sum().sum()} missing values")
print(f"After encoding: {X_encoded.shape}, dtype: {X_encoded.dtypes.unique()}")
```

### 4. Validate Predictions

```python
# In prediction
assert np.all((y_pred >= 0) & (y_pred <= 1)), "Predictions outside [0, 1]"
assert not np.any(np.isnan(y_pred)), "NaN values in predictions"
```

## Performance Optimization

### 1. Parallelize Cross-Validation

```yaml
training:
  cv_n_jobs: -1  # Use all cores
```

### 2. Use Sparse Matrices for OneHot Encoding

```python
# In encoding
X_encoded = encoder.transform(X, sparse=True)
```

### 3. Cache Expensive Operations

```python
# Store preprocessed data
X_train_imputed_cached = X_train_imputed.copy()
```

## Common Pitfalls

### ❌ Mistake: Data Leakage

```python
# BAD: Fit imputation on entire dataset
imputer.fit(X)  # Includes test data!

# GOOD: Fit on training data only
imputer.fit(X_train)
X_train_clean = imputer.transform(X_train)
X_test_clean = imputer.transform(X_test)
```

### ❌ Mistake: Mixing Probability and Class Labels

```python
# BAD: Using class labels instead of probabilities
y_pred_labels = np.argmax(y_pred_proba, axis=1)
auroc = roc_auc_score(y_true, y_pred_labels)  # Wrong!

# GOOD: Use probabilities for ROC AUC
auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
```

### ❌ Mistake: Not Handling Binary Probabilities

```python
# BAD: Treating as multiclass
y_pred_proba  # Shape: (n, 2)

# GOOD: Extract column 1 for binary ROC AUC
auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
```

## Reference

- **SYSTEM_DESIGN.md** - Full system architecture
- **architecture.md** - Component relationships and data flow
- **PROBLEM_DESCRIPTION.md** - Competition details and data schema
- **docs/build-scaffolds/** - Implementation checklists and plans

---

**Last Updated**: January 7, 2026  
**Scaffolding Version**: 1.0
