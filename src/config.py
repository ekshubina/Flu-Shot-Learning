"""
Configuration system for the ML pipeline.

This module defines dataclasses for all configuration options, enabling type-safe,
validated parameter passing throughout the pipeline. Configurations can be loaded
from dictionaries or YAML files and composed into a top-level PipelineConfig.

See SYSTEM_DESIGN.md for architecture overview and expected parameter values.
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json


# Feature type constants for type-based imputation
# These are based on the problem definition and ordinal/nominal semantics

ORDINAL_COLUMNS = [
    # Concern & Knowledge (2)
    "h1n1_concern",
    "h1n1_knowledge",
    # Opinions (6) - All Likert scale 1-5
    "opinion_h1n1_vacc_effective",
    "opinion_h1n1_risk",
    "opinion_h1n1_sick_from_vacc",
    "opinion_seas_vacc_effective",
    "opinion_seas_risk",
    "opinion_seas_sick_from_vacc",
    # Ordinal demographics (3) - have natural order
    "age_group",
    "education",
    "income_poverty",
]

NOMINAL_COLUMNS = [
    # Categorical demographics and employment (9)
    "race",
    "sex",
    "marital_status",
    "rent_or_own",
    "employment_status",
    "hhs_geo_region",
    "census_msa",
    "employment_industry",
    "employment_occupation",
]

BINARY_NUMERIC_COLUMNS = [
    # Behavioral (7)
    "behavioral_antiviral_meds",
    "behavioral_avoidance",
    "behavioral_face_mask",
    "behavioral_wash_hands",
    "behavioral_large_gatherings",
    "behavioral_outside_home",
    "behavioral_touch_face",
    # Medical recommendations & status (5)
    "doctor_recc_h1n1",
    "doctor_recc_seasonal",
    "chronic_med_condition",
    "child_under_6_months",
    "health_worker",
    # Other (1)
    "health_insurance",
    # Household (2)
    "household_adults",
    "household_children",
]


@dataclass
class DataConfig:
    """
    Configuration for data loading and splitting.
    
    Attributes:
        train_features_path: Path to training features CSV
        train_labels_path: Path to training labels CSV
        test_features_path: Path to test features CSV
        submission_format_path: Path to submission format template CSV
        data_dir: Root directory containing data files (used if paths are relative)
        cv_folds: Number of cross-validation folds (default: 5)
        random_seed: Random seed for reproducibility (default: 42)
        stratify: Whether to use stratified k-fold (default: True)
        test_size: Test/validation set fraction for holdout validation (default: 0.2)
    """
    train_features_path: str = "data/training_set_features.csv"
    train_labels_path: str = "data/training_set_labels.csv"
    test_features_path: str = "data/test_set_features.csv"
    submission_format_path: str = "data/submission_format.csv"
    data_dir: Optional[str] = None
    cv_folds: int = 5
    random_seed: int = 42
    stratify: bool = True
    test_size: float = 0.2


@dataclass
class ImputationConfig:
    """
    Configuration for missing value imputation strategy (single strategy for all columns).
    
    Attributes:
        strategy: Name of imputation strategy to use
                 (drop_rows, drop_columns, mean, mode, knn, mice, flag_as_missing)
        n_neighbors: For KNN imputation, number of neighbors (default: 5)
        mice_iterations: For MICE, number of iterations (default: 10)
        fill_value: For constant fill, value to use (default: 0)
        drop_threshold: Drop columns with missing % above this threshold (default: 0.5)
    """
    strategy: str = "mean"
    n_neighbors: int = 5
    mice_iterations: int = 10
    fill_value: float = 0.0
    drop_threshold: float = 0.5


@dataclass
class TypeBasedImputationConfig:
    """
    Configuration for type-based imputation with different strategies per feature type.
    
    Applies different imputation strategies to different feature types:
    - Ordinal features: mean, median, knn, or mice
    - Nominal features: mode or mice
    - Binary numeric features: treated as ordinal (mean, median, knn, or mice)
    
    This enables flexible strategy combinations, e.g.:
    - ordinal_strategy='mean', nominal_strategy='mode' (default)
    - ordinal_strategy='knn', nominal_strategy='mode' (advanced)
    - ordinal_strategy='mice', nominal_strategy='mice' (multivariate)
    
    Attributes:
        type: Strategy type (always 'type_based' for this config)
        ordinal_strategy: Strategy for ordinal and binary numeric columns
                         (mean, median, knn, mice; default: 'mean')
        nominal_strategy: Strategy for nominal columns
                         (mode, mice; default: 'mode')
        ordinal_params: Dict of parameters for ordinal strategy
                       (e.g., {'n_neighbors': 5} for knn; default: {})
        nominal_params: Dict of parameters for nominal strategy
                       (e.g., {} for mode; default: {})
    """
    type: str = "type_based"
    ordinal_strategy: str = "mean"
    nominal_strategy: str = "mode"
    ordinal_params: Dict[str, Any] = field(default_factory=dict)
    nominal_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Validate configuration at initialization time.
        
        Raises:
            ValueError: If strategy combination is invalid
        """
        valid_ordinal_strategies = {'mean', 'median', 'knn', 'mice'}
        valid_nominal_strategies = {'mode', 'mice'}
        
        if self.ordinal_strategy not in valid_ordinal_strategies:
            raise ValueError(
                f"Invalid ordinal_strategy '{self.ordinal_strategy}'. "
                f"Must be one of: {sorted(valid_ordinal_strategies)}"
            )
        
        if self.nominal_strategy not in valid_nominal_strategies:
            raise ValueError(
                f"Invalid nominal_strategy '{self.nominal_strategy}'. "
                f"KNN imputation is not suitable for nominal features. "
                f"Must be one of: {sorted(valid_nominal_strategies)}"
            )


@dataclass
class EncodingConfig:
    """
    Configuration for feature encoding strategies.
    
    Attributes:
        strategies: Dictionary mapping feature groups to encoding strategies
        ordinal_features: List of feature names to encode as ordinal (preserve order)
        categorical_features: List of feature names to encode categorically
        binary_features: List of binary features (already 0/1, no encoding needed)
        ordinal_encoding_type: Method for ordinal features (ordinal, label, as_is)
        categorical_encoding_type: Method for categorical features (one_hot, target, ordinal)
        interaction_terms: Whether to create interaction terms (default: False)
        polynomial_degree: Degree for polynomial features (default: 2, 0 to disable)
        drop_first_onehot: Drop first category in one-hot to avoid multicollinearity (default: True)
        target_encoding_smoothing: Smoothing parameter for target encoding (default: 1.0)
        drop_features: List of features to drop before encoding (default: [])
    """
    strategies: Dict[str, Any] = field(default_factory=dict)
    ordinal_features: List[str] = field(default_factory=lambda: [
        "h1n1_concern", "h1n1_knowledge",
        "opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
        "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
        "opinion_seas_risk", "opinion_seas_sick_from_vacc"
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        "age_group", "education", "race", "sex", "income_poverty",
        "marital_status", "rent_or_own", "employment_status",
        "hhs_geo_region", "census_msa", "employment_industry", "employment_occupation"
    ])
    binary_features: List[str] = field(default_factory=lambda: [
        "behavioral_antiviral_meds", "behavioral_avoidance",
        "behavioral_face_mask", "behavioral_large_gatherings",
        "behavioral_outside_home", "behavioral_touch_face",
        "doctor_recc_h1n1", "doctor_recc_seasonal",
        "chronic_med_condition", "health_worker", "health_insurance",
        "household_adults", "household_children"
    ])
    ordinal_encoding_type: str = "ordinal"
    categorical_encoding_type: str = "one_hot"
    interaction_terms: bool = False
    polynomial_degree: int = 0
    drop_first_onehot: bool = True
    target_encoding_smoothing: float = 1.0
    drop_features: List[str] = field(default_factory=list)


@dataclass
class FeatureEngineeringConfig:
    """
    Configuration for feature engineering operations.
    
    Feature engineering applies transformations to raw features after imputation
    but before encoding. Operations include:
    - Missing value indicators: Create binary flags for originally-missing values
    - Employment simplification: Consolidate employment features
    - Interaction terms: Create cross-product features
    - Polynomial features: Create polynomial transformations
    
    Attributes:
        missing_flags: Whether to create binary indicators for missing values (default: False)
        missing_flag_features: List of features to create missing indicators for (default: [])
        simplify_employment: Whether to simplify employment features (default: False)
        create_interactions: Whether to create interaction terms (default: False)
        interaction_pairs: List of (feature1, feature2) pairs for interactions (default: [])
        create_polynomials: Whether to create polynomial features (default: False)
        polynomial_degree: Degree of polynomial features (default: 2)
    """
    missing_flags: bool = False
    missing_flag_features: List[str] = field(default_factory=list)
    simplify_employment: bool = False
    create_interactions: bool = False
    interaction_pairs: List[tuple] = field(default_factory=list)
    create_polynomials: bool = False
    polynomial_degree: int = 2


@dataclass
class ModelConfig:
    """
    Configuration for model selection and hyperparameters.
    
    Attributes:
        model_type: Type of model to use
                   (logistic_regression, xgboost, lightgbm, random_forest, gradient_boosting)
        hyperparameters: Dictionary of model-specific hyperparameters
        random_seed: Random seed for model reproducibility (default: 42)
        n_jobs: Number of parallel jobs (-1 for all cores, default: 1)
        class_weight: How to weight classes in loss (None, balanced, custom dict)
        sample_weight: Whether to use custom sample weights (default: False)
    """
    model_type: str = "logistic_regression"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    n_jobs: int = 1
    class_weight: Optional[str] = None
    sample_weight: bool = False


@dataclass
class TrainingConfig:
    """
    Configuration for model training strategy.
    
    Attributes:
        cv_strategy: Cross-validation strategy (stratified_kfold, kfold, time_series)
        cv_folds: Number of cross-validation folds (default: 5)
        test_size: Test set fraction for each fold (default: 0.2)
        use_smote: Whether to apply SMOTE for class imbalance (default: False)
        smote_ratio: Ratio of minority to majority class after SMOTE (default: 0.5)
        threshold_tuning: Whether to tune decision threshold on validation set (default: False)
        threshold_metric: Metric to optimize threshold for (auc, f1, precision, recall)
        early_stopping: Whether to use early stopping (for tree-based models, default: False)
        early_stopping_rounds: Number of rounds without improvement before stopping (default: 10)
        hyperparameter_search: Whether to perform hyperparameter search (default: False)
        search_strategy: Hyperparameter search strategy (grid, random, bayesian)
        search_cv_folds: CV folds for hyperparameter search (default: 3)
        class_weight_strategy: Class weight strategy (default: balanced)
    """
    cv_strategy: str = "stratified_kfold"
    cv_folds: int = 5
    test_size: float = 0.2
    use_smote: bool = False
    smote_ratio: float = 0.5
    threshold_tuning: bool = False
    threshold_metric: str = "auc"
    early_stopping: bool = False
    early_stopping_rounds: int = 10
    hyperparameter_search: bool = False
    search_strategy: str = "grid"
    search_cv_folds: int = 3
    class_weight_strategy: str = "balanced"


@dataclass
class CalibrationConfig:
    """
    Configuration for prediction calibration.
    
    Attributes:
        method: Calibration method to use
               (none, platt_scaling, isotonic, temperature_scaling)
        calibration_cv_folds: Number of folds for calibration fitting (default: 5)
        calibration_folds: Alternative name for calibration_cv_folds (from YAML)
        smooth_calibration: Whether to use smoothing in isotonic calibration (default: False)
    """
    method: str = "none"
    calibration_cv_folds: int = 5
    calibration_folds: Optional[int] = None
    smooth_calibration: bool = False
    
    def __post_init__(self):
        """Normalize field names (handle both calibration_folds and calibration_cv_folds)."""
        if self.calibration_folds is not None and self.calibration_cv_folds == 5:
            self.calibration_cv_folds = self.calibration_folds
        elif self.calibration_folds is None and self.calibration_cv_folds != 5:
            self.calibration_folds = self.calibration_cv_folds


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation and metrics.
    
    Attributes:
        metrics: List of metrics to compute (auroc, accuracy, f1, precision, recall, roc_auc_per_vaccine)
        compute_calibration_error: Whether to compute calibration error metrics (ECE, MCE)
        plot_roc_curves: Whether to generate ROC curve plots (default: True)
        plot_calibration_curves: Whether to generate calibration plots (default: True)
        plot_feature_importance: Whether to plot feature importance (default: True)
        output_dir: Directory to save plots and detailed results (default: "results/")
    """
    metrics: List[str] = field(default_factory=lambda: [
        "auroc", "accuracy", "f1", "precision", "recall"
    ])
    compute_calibration_error: bool = True
    plot_roc_curves: bool = True
    plot_calibration_curves: bool = True
    plot_feature_importance: bool = True
    output_dir: str = "results/"


@dataclass
class PredictionConfig:
    """
    Configuration for test set prediction and submission generation.
    
    Attributes:
        output_dir: Directory to save submission CSV (default: "./submissions/")
        output_filename: Filename for submission CSV (default: "submission_baseline.csv")
        validate_submission: Whether to validate submission format (default: True)
    """
    output_dir: str = "./submissions/"
    output_filename: str = "submission_baseline.csv"
    validate_submission: bool = True


@dataclass
class LoggingConfig:
    """
    Configuration for logging throughout the pipeline.
    
    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, default: INFO)
        log_file: Path to log file (default: "logs/pipeline.log")
        format: Log format string (default: standard format)
    """
    level: str = "INFO"
    log_file: str = "logs/pipeline.log"
    format: str = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"


@dataclass
class TrackingConfig:
    """
    Configuration for experiment tracking and logging.
    
    Attributes:
        enabled: Whether tracking is enabled (default: True)
        tracker_type: Type of experiment tracker (csv, mlflow, wandb, none)
        log_dir: Directory to store experiment logs (default: "logs/")
        log_path: Alternative name for log file path (from YAML)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path (default: "logs/pipeline.log")
        log_to_file: Whether to log to file in addition to console (default: True)
        log_frequency: How often to log during training (default: 100)
        format: Log format string (default: standard format)
        track_hyperparameters: Whether to log hyperparameters (default: True)
        track_metrics: Whether to log metrics (default: True)
        track_data_summary: Whether to log data summaries (default: True)
        log_metrics: List of specific metrics to track (default: [])
    """
    enabled: bool = True
    tracker_type: str = "csv"
    log_dir: str = "logs/"
    log_path: Optional[str] = None
    log_level: str = "INFO"
    log_file: str = "logs/pipeline.log"
    log_to_file: bool = True
    log_frequency: int = 100
    format: str = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    track_hyperparameters: bool = True
    track_metrics: bool = True
    track_data_summary: bool = True
    log_metrics: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """
    Top-level configuration composing all pipeline stage configs.
    
    Attributes:
        name: Name/description of this pipeline configuration
        description: Longer description of the pipeline experiment
        data: DataConfig instance
        imputation: ImputationConfig instance
        feature_engineering: FeatureEngineeringConfig instance
        encoding: EncodingConfig instance
        model: ModelConfig instance
        training: TrainingConfig instance
        calibration: CalibrationConfig instance
        evaluation: EvaluationConfig instance
        prediction: PredictionConfig instance
        tracking: TrackingConfig instance
        logging: LoggingConfig instance
    """
    name: str = "default_pipeline"
    description: str = "Default ML pipeline configuration"
    data: DataConfig = field(default_factory=DataConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """
        Create PipelineConfig from dictionary.
        
        Args:
            config_dict: Dictionary with keys matching PipelineConfig attributes
            
        Returns:
            PipelineConfig instance
        """
        # Extract sub-configs or use defaults
        data_dict = config_dict.get("data", {})
        imputation_dict = config_dict.get("imputation", {})
        feature_engineering_dict = config_dict.get("feature_engineering", {})
        encoding_dict = config_dict.get("encoding", {})
        model_dict = config_dict.get("model", {})
        training_dict = config_dict.get("training", {})
        calibration_dict = config_dict.get("calibration", {})
        evaluation_dict = config_dict.get("evaluation", {})
        prediction_dict = config_dict.get("prediction", {})
        tracking_dict = config_dict.get("tracking", {})
        logging_dict = config_dict.get("logging", {})
        
        # Helper function to filter out unexpected kwargs
        def filter_dict(d, expected_fields):
            return {k: v for k, v in d.items() if k in expected_fields}
        
        # Determine which imputation config class to use based on 'type' field
        if imputation_dict:
            imputation_type = imputation_dict.get("type")
            if imputation_type == "type_based":
                # Use TypeBasedImputationConfig for type-based imputation
                imputation_config = TypeBasedImputationConfig(
                    **filter_dict(imputation_dict, {f.name for f in TypeBasedImputationConfig.__dataclass_fields__.values()})
                )
            else:
                # Use ImputationConfig for single-strategy imputation (default)
                imputation_config = ImputationConfig(
                    **filter_dict(imputation_dict, {f.name for f in ImputationConfig.__dataclass_fields__.values()})
                )
        else:
            imputation_config = ImputationConfig()
        
        return cls(
            name=config_dict.get("name", "default_pipeline"),
            description=config_dict.get("description", "Default ML pipeline configuration"),
            data=DataConfig(**filter_dict(data_dict, {f.name for f in DataConfig.__dataclass_fields__.values()})) if data_dict else DataConfig(),
            imputation=imputation_config,
            feature_engineering=FeatureEngineeringConfig(**filter_dict(feature_engineering_dict, {f.name for f in FeatureEngineeringConfig.__dataclass_fields__.values()})) if feature_engineering_dict else FeatureEngineeringConfig(),
            encoding=EncodingConfig(**filter_dict(encoding_dict, {f.name for f in EncodingConfig.__dataclass_fields__.values()})) if encoding_dict else EncodingConfig(),
            model=ModelConfig(**filter_dict(model_dict, {f.name for f in ModelConfig.__dataclass_fields__.values()})) if model_dict else ModelConfig(),
            training=TrainingConfig(**filter_dict(training_dict, {f.name for f in TrainingConfig.__dataclass_fields__.values()})) if training_dict else TrainingConfig(),
            calibration=CalibrationConfig(**filter_dict(calibration_dict, {f.name for f in CalibrationConfig.__dataclass_fields__.values()})) if calibration_dict else CalibrationConfig(),
            evaluation=EvaluationConfig(**filter_dict(evaluation_dict, {f.name for f in EvaluationConfig.__dataclass_fields__.values()})) if evaluation_dict else EvaluationConfig(),
            prediction=PredictionConfig(**filter_dict(prediction_dict, {f.name for f in PredictionConfig.__dataclass_fields__.values()})) if prediction_dict else PredictionConfig(),
            tracking=TrackingConfig(**filter_dict(tracking_dict, {f.name for f in TrackingConfig.__dataclass_fields__.values()})) if tracking_dict else TrackingConfig(),
            logging=LoggingConfig(**filter_dict(logging_dict, {f.name for f in LoggingConfig.__dataclass_fields__.values()})) if logging_dict else LoggingConfig(),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PipelineConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            PipelineConfig instance
            
        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If YAML file does not exist
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load config from YAML. Install with: pip install pyyaml")
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "PipelineConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            PipelineConfig instance
            
        Raises:
            FileNotFoundError: If JSON file does not exist
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
            
        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save config to YAML. Install with: pip install pyyaml")
        
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_json(self, json_path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration file
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """
    Load configuration from file (auto-detects format from extension).
    
    Args:
        config_path: Path to configuration file (.yaml, .yml, .json)
        
    Returns:
        PipelineConfig instance
        
    Raises:
        ValueError: If file format is not supported
    """
    path = Path(config_path)
    
    if path.suffix in ['.yaml', '.yml']:
        return PipelineConfig.from_yaml(config_path)
    elif path.suffix == '.json':
        return PipelineConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration format: {path.suffix}. Use .yaml, .yml, or .json")


# Feature group constants for reference in encoding and analysis
FEATURE_GROUPS = {
    "opinions": [
        "opinion_h1n1_vacc_effective",
        "opinion_h1n1_risk",
        "opinion_h1n1_sick_from_vacc",
        "opinion_seas_vacc_effective",
        "opinion_seas_risk",
        "opinion_seas_sick_from_vacc",
    ],
    "behavioral": [
        "behavioral_antiviral_meds",
        "behavioral_avoidance",
        "behavioral_face_mask",
        "behavioral_large_gatherings",
        "behavioral_outside_home",
        "behavioral_touch_face",
    ],
    "medical": [
        "doctor_recc_h1n1",
        "doctor_recc_seasonal",
        "chronic_med_condition",
        "health_worker",
        "health_insurance",
    ],
    "demographics": [
        "age_group",
        "education",
        "race",
        "sex",
        "income_poverty",
        "marital_status",
        "rent_or_own",
        "employment_status",
    ],
    "household": [
        "household_adults",
        "household_children",
    ],
    "geographic": [
        "hhs_geo_region",
        "census_msa",
    ],
    "concern_knowledge": [
        "h1n1_concern",
        "h1n1_knowledge",
    ],
    "employment": [
        "employment_industry",
        "employment_occupation",
    ],
}

# Target variables
TARGETS = ["h1n1_vaccine", "seasonal_vaccine"]

# ID column
ID_COLUMN = "respondent_id"
