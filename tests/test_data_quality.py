import pytest
import pandas as pd
import numpy as np
from src.data_ingestion import get_initial_audit
from src.data_quality import generate_quality_report, impute_or_clean_column, validate_and_cast_type

def test_initial_audit():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": ["x", "x", "x", "x", "x"],
        "c": [np.nan, 1, 2, 3, 4]
    })
    audit = get_initial_audit(df)
    assert audit["n_rows"] == 5
    assert audit["n_cols"] == 3
    assert "b" in audit["constant_cols"]
    assert audit["total_missing_cells"] == 1

def test_quality_report():
    df = pd.DataFrame({
        "age": [25, 30, 35, 1000],
        "city": ["NYC", "LA", None, "NYC"]
    })
    report = generate_quality_report(df)
    assert len(report) == 2
    assert "Variable" in report.columns
    assert report.loc[report["Variable"] == "city", "Missing"].iloc[0] == 1

def test_imputation_and_casting():
    df = pd.DataFrame({"val": [10.0, 20.0, np.nan]})
    df_imputed, msg = impute_or_clean_column(df, "val", "impute_mean")
    assert df_imputed["val"].isna().sum() == 0
    assert df_imputed["val"].iloc[2] == 15.0

    df_str = pd.DataFrame({"num_str": ["10", "20", "30"]})
    df_cast, msg, success = validate_and_cast_type(df_str, "num_str", "Numeric")
    assert success
    assert np.issubdtype(df_cast["num_str"].dtype, np.number)
