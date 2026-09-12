import pytest
import pandas as pd
import numpy as np
from src.preprocessing import validate_target_variable, perform_stratified_split, build_preprocessing_pipeline

def test_validate_target():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    res = validate_target_variable(df, "target")
    assert res["problem_type"] == "binary"
    assert res["n_unique"] == 2
    assert not res["is_imbalanced"]

def test_perform_split():
    df = pd.DataFrame({
        "f1": range(20),
        "target": [0, 1] * 10
    })
    X_tr, X_te, y_tr, y_te = perform_stratified_split(df, "target", test_size=0.2, random_state=42)
    assert len(X_tr) == 16
    assert len(X_te) == 4
    assert len(y_tr) == 16
    assert len(y_te) == 4

def test_pipeline_building():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["A", "B", "A", "B"]
    })
    preprocessor = build_preprocessing_pipeline(df, ["num"], ["cat"])
    X_trans = preprocessor.fit_transform(df)
    assert X_trans.shape[0] == 4
    assert X_trans.shape[1] == 3 # 1 scaled num + 2 one-hot cats
