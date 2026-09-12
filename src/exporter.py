import io
import json
import joblib
import pandas as pd
from typing import Dict, Any

def export_dataframe_to_bytes(df: pd.DataFrame, format_type: str = "csv") -> bytes:
    """
    Converts DataFrame to bytes for download in Streamlit (CSV or Excel).
    """
    if format_type == "csv":
        return df.to_csv(index=False).encode('utf-8')
    elif format_type == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()
    else:
        raise ValueError(f"Format {format_type} not supported.")

def export_text_to_bytes(text_content: str) -> bytes:
    """
    Converts string content to bytes for text/markdown downloads.
    """
    return text_content.encode('utf-8')

def export_model_to_bytes(model_pipeline) -> bytes:
    """
    Serializes trained scikit-learn model pipeline to joblib pickle bytes.
    """
    output = io.BytesIO()
    joblib.dump(model_pipeline, output)
    return output.getvalue()

def export_dict_to_json_bytes(data_dict: Dict[str, Any]) -> bytes:
    """
    Serializes python dictionary to JSON bytes.
    """
    # Clean non-serializable objects
    cleaned_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, (int, float, str, bool, list, dict)):
            cleaned_dict[k] = v
        else:
            cleaned_dict[k] = str(v)
    return json.dumps(cleaned_dict, indent=2).encode('utf-8')
