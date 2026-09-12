import pandas as pd
import numpy as np
import io

def load_data(uploaded_file, file_type: str, delimiter: str = ",", sheet_name: str = 0) -> pd.DataFrame:
    """
    Loads uploaded data file into a pandas DataFrame.
    Supports CSV, TXT, XLSX, XLS.
    """
    if file_type in ["csv", "txt"]:
        # Try automatic fallback delimiters if comma fails
        try:
            df = pd.read_csv(uploaded_file, sep=delimiter)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
    elif file_type in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file format: {file_type}")
    return df

def get_initial_audit(df: pd.DataFrame) -> dict:
    """
    Performs initial data audit detecting:
    - row and column count
    - missing values per column
    - duplicate rows
    - constant or near-constant columns
    - high cardinality text columns
    - potential numeric variables stored as string/text
    """
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    
    # Near constant (single value dominates > 98%)
    near_constant_cols = []
    for col in df.columns:
        if col not in constant_cols and len(df) > 0:
            top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq >= 0.98:
                near_constant_cols.append(col)
                
    high_cardinality_cols = []
    text_as_numeric_cols = []
    
    for col in df.columns:
        n_unique = df[col].nunique()
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            # High cardinality if unique text > 50% of dataset or > 500 unique values
            if n_unique > 0.5 * n_rows or n_unique > 500:
                high_cardinality_cols.append(col)
            # Check if object can be converted to numeric
            try:
                converted = pd.to_numeric(df[col].dropna(), errors='coerce')
                if converted.notna().all() and len(converted) > 0:
                    text_as_numeric_cols.append(col)
            except Exception:
                pass
                
    missing_summary = df.isnull().sum()
    total_missing_cells = int(missing_summary.sum())
    missing_pct_overall = float((total_missing_cells / (n_rows * n_cols)) * 100) if n_rows * n_cols > 0 else 0.0
    
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": duplicate_rows,
        "total_missing_cells": total_missing_cells,
        "missing_pct_overall": round(missing_pct_overall, 2),
        "constant_cols": constant_cols,
        "near_constant_cols": near_constant_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "text_as_numeric_cols": text_as_numeric_cols
    }
