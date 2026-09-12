import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Any, List, Optional
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
import shap

def extract_feature_names_from_preprocessor(preprocessor, feature_names_in: List[str]) -> List[str]:
    """
    Extracts transformed feature names from ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        return feature_names_in

def get_tree_feature_importances(model_pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts tree feature importances (Gini/MDI) from fitted model pipeline.
    """
    model = model_pipeline.named_steps.get('classifier', model_pipeline[-1])
    
    if not hasattr(model, 'feature_importances_'):
        return pd.DataFrame()
        
    importances = model.feature_importances_
    
    # Align length if feature names match
    if len(feature_names) == len(importances):
        cols = feature_names
    else:
        cols = [f"Feature_{i}" for i in range(len(importances))]
        
    df_imp = pd.DataFrame({
        "Feature": cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    return df_imp

def calculate_permutation_importance(
    model_pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    scoring: str = "f1_weighted",
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes permutation importance on Test Set.
    """
    result = permutation_importance(
        model_pipeline, X_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=random_state
    )
    
    df_perm = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std
    }).sort_values(by="Importance_Mean", ascending=False)
    
    return df_perm

def plot_feature_importance_bar(df_imp: pd.DataFrame, title: str = "Top Feature Importances", top_n: int = 15):
    """
    Plots horizontal bar chart of feature importances.
    """
    df_top = df_imp.head(top_n).sort_values(by=df_imp.columns[1], ascending=True)
    val_col = df_imp.columns[1]
    
    fig = px.bar(
        df_top, x=val_col, y="Feature", orientation="h", text_auto=".3f",
        title=title, color=val_col, color_continuous_scale="Viridis"
    )
    fig.update_layout(template="plotly_white", title_x=0.5, height=450)
    return fig

def get_decision_tree_rules(model_pipeline, feature_names: List[str]) -> str:
    """
    Extracts text representation of rules for DecisionTreeClassifier.
    """
    model = model_pipeline.named_steps.get('classifier', model_pipeline[-1])
    if hasattr(model, 'tree_'):
        return export_text(model, feature_names=feature_names, max_depth=4)
    return "Rule extraction is only applicable for single Decision Tree models."
