import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

def validate_target_variable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Validates target suitability, auto-detects binary vs multiclass,
    checks for missing target values, severe imbalance, high cardinality, and potential leakage.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        
    y_series = df[target_col].dropna()
    n_missing = int(df[target_col].isna().sum())
    unique_vals = y_series.unique()
    n_unique = len(unique_vals)
    
    if n_unique <= 1:
        problem_type = "invalid"
    elif n_unique == 2:
        problem_type = "binary"
    else:
        problem_type = "multiclass"
        
    # Check class imbalance
    class_counts = y_series.value_counts(normalize=True)
    min_class_pct = float(class_counts.min() * 100)
    is_imbalanced = min_class_pct < 15.0 # Warning threshold < 15%
    
    # Check potential leakage predictors (e.g. 100% correlation with target or identical values)
    leakage_suspects = []
    for col in df.columns:
        if col != target_col:
            if df[col].nunique() == n_unique and df[col].equals(df[target_col]):
                leakage_suspects.append(col)
                
    warnings = []
    if n_missing > 0:
        warnings.append(f"Target variable has {n_missing} missing values. Rows with missing target must be dropped.")
    if n_unique > 20:
        warnings.append(f"Target variable has {n_unique} unique categories. Ensure this is a classification problem, not regression.")
    if is_imbalanced:
        warnings.append(f"Minority class represents only {min_class_pct:.1f}% of data. Consider stratified splitting or evaluation metrics like Recall/PR-AUC.")
    if leakage_suspects:
        warnings.append(f"Potential target leakage detected in features: {leakage_suspects}. Consider removing these columns.")
        
    return {
        "problem_type": problem_type,
        "n_unique": n_unique,
        "n_missing": n_missing,
        "class_counts": class_counts.to_dict(),
        "is_imbalanced": is_imbalanced,
        "min_class_pct": min_class_pct,
        "leakage_suspects": leakage_suspects,
        "warnings": warnings
    }

def perform_stratified_split(
    df: pd.DataFrame, 
    target_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans target NAs and performs stratified train/test split.
    Returns (X_train, X_test, y_train, y_test)
    """
    clean_df = df.dropna(subset=[target_col]).copy()
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]
    
    # Try stratified split; fallback to standard split if class count is too small
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(
    X_train: pd.DataFrame, 
    numerical_cols: List[str], 
    categorical_cols: List[str]
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Builds a scikit-learn ColumnTransformer for numerical & categorical features.
    Strictly fitted on X_train to prevent data leakage.
    Returns (preprocessor, transformed_feature_names)
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor
