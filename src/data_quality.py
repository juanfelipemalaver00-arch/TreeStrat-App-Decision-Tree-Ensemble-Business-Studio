import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

def generate_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a comprehensive Data Quality summary table per column.
    """
    report_rows = []
    n_rows = len(df)
    
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        missing_pct = round((n_missing / n_rows) * 100, 2) if n_rows > 0 else 0.0
        n_unique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)
        
        # Outlier calculation for numeric columns using IQR
        n_outliers = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_vals = df[col].dropna()
            if len(valid_vals) > 0:
                q25, q75 = valid_vals.quantile(0.25), valid_vals.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                n_outliers = int(((valid_vals < lower_bound) | (valid_vals > upper_bound)).sum())
                
        # Generate Smart Recommendation
        if n_unique <= 1:
            rec = "Drop Variable (Constant)"
        elif n_unique == n_rows and dtype_str in ['object', 'string']:
            rec = "Drop Variable (ID / High Cardinality)"
        elif missing_pct > 50.0:
            rec = "Consider Dropping (High Missingness)"
        elif missing_pct > 0.0:
            if pd.api.types.is_numeric_dtype(df[col]):
                rec = "Impute Median / Mean"
            else:
                rec = "Impute Mode or 'Missing' Category"
        elif n_outliers > 0:
            rec = f"Inspect {n_outliers} Outliers"
        else:
            rec = "Keep / No Action Needed"
            
        report_rows.append({
            "Variable": col,
            "Type": dtype_str,
            "Missing": n_missing,
            "Missing %": missing_pct,
            "Unique Values": n_unique,
            "Outliers (IQR)": n_outliers,
            "Recommendation": rec
        })
        
    return pd.DataFrame(report_rows)

def impute_or_clean_column(df: pd.DataFrame, col: str, strategy: str, custom_val: Any = None) -> Tuple[pd.DataFrame, str]:
    """
    Applies data quality cleaning action on a specific column.
    Strategies:
    - 'drop_rows': Remove rows where col is missing
    - 'drop_col': Remove entire column
    - 'impute_mean': Fill missing numeric with mean
    - 'impute_median': Fill missing numeric with median
    - 'impute_mode': Fill missing with most frequent value
    - 'impute_category': Fill missing categorical with 'Missing' label
    - 'impute_custom': Fill missing with custom value
    """
    df_clean = df.copy()
    msg = ""
    
    if strategy == "drop_rows":
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=[col])
        after = len(df_clean)
        msg = f"Dropped {before - after} rows with missing values in '{col}'."
    elif strategy == "drop_col":
        df_clean = df_clean.drop(columns=[col])
        msg = f"Dropped column '{col}' from dataset."
    elif strategy == "impute_mean":
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            val = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(val)
            msg = f"Imputed missing values in '{col}' with mean ({val:.2f})."
        else:
            msg = f"Error: Cannot compute mean for non-numeric column '{col}'."
    elif strategy == "impute_median":
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(val)
            msg = f"Imputed missing values in '{col}' with median ({val:.2f})."
        else:
            msg = f"Error: Cannot compute median for non-numeric column '{col}'."
    elif strategy == "impute_mode":
        mode_val = df_clean[col].mode()
        val = mode_val.iloc[0] if len(mode_val) > 0 else "Missing"
        df_clean[col] = df_clean[col].fillna(val)
        msg = f"Imputed missing values in '{col}' with mode ('{val}')."
    elif strategy == "impute_category":
        df_clean[col] = df_clean[col].fillna("Missing").astype(str)
        msg = f"Imputed missing values in '{col}' with category 'Missing'."
    elif strategy == "impute_custom" and custom_val is not None:
        df_clean[col] = df_clean[col].fillna(custom_val)
        msg = f"Imputed missing values in '{col}' with custom value '{custom_val}'."
        
    return df_clean, msg

def validate_and_cast_type(df: pd.DataFrame, col: str, target_type: str) -> Tuple[pd.DataFrame, str, bool]:
    """
    Safely changes the data type of a column.
    Returns (modified_df, warning_message, is_success)
    """
    df_cast = df.copy()
    original_series = df_cast[col]
    
    try:
        if target_type == "Numeric":
            converted = pd.to_numeric(original_series, errors='coerce')
            new_nans = converted.isna().sum() - original_series.isna().sum()
            if new_nans > 0:
                warning = f"Warning: Converting '{col}' to Numeric generated {new_nans} NaN values from unparseable strings."
            else:
                warning = f"Successfully converted '{col}' to Numeric."
            df_cast[col] = converted
            return df_cast, warning, True
            
        elif target_type == "Categorical":
            df_cast[col] = original_series.astype(str).astype('category')
            return df_cast, f"Successfully converted '{col}' to Categorical.", True
            
        elif target_type == "Integer":
            converted = pd.to_numeric(original_series, errors='coerce').round()
            df_cast[col] = converted.astype('Int64') # Supports NA integers
            return df_cast, f"Successfully converted '{col}' to Integer.", True
            
        elif target_type == "Date":
            converted = pd.to_datetime(original_series, errors='coerce')
            new_nans = converted.isna().sum() - original_series.isna().sum()
            if new_nans > 0:
                warning = f"Warning: Converting '{col}' to Date generated {new_nans} NaN values."
            else:
                warning = f"Successfully converted '{col}' to Datetime."
            df_cast[col] = converted
            return df_cast, warning, True
            
        elif target_type == "String / Text":
            df_cast[col] = original_series.astype(str)
            return df_cast, f"Successfully converted '{col}' to String.", True
            
    except Exception as e:
        return df, f"Error converting '{col}' to {target_type}: {str(e)}", False
        
    return df, "No type change applied.", True
