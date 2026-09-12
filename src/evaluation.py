import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, confusion_matrix, precision_recall_curve, auc
)

def evaluate_model_performance(
    model_pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    problem_type: str = "binary",
    fn_cost_weight: float = 3.0,
    fp_cost_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluates a trained model pipeline on reserved Test data.
    Computes standard ML metrics and custom business cost-weighted score.
    """
    y_pred = model_pipeline.predict(X_test)
    
    # Check if model supports predict_proba
    y_proba = None
    if hasattr(model_pipeline, "predict_proba"):
        try:
            y_proba = model_pipeline.predict_proba(X_test)
        except Exception:
            pass
            
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    average_type = "binary" if problem_type == "binary" else "weighted"
    prec = precision_score(y_test, y_pred, average=average_type, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average_type, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_type, zero_division=0)
    
    roc_auc_val = None
    pr_auc_val = None
    log_loss_val = None
    
    if y_proba is not None:
        try:
            if problem_type == "binary":
                positive_proba = y_proba[:, 1]
                roc_auc_val = roc_auc_score(y_test, positive_proba)
                p_curve, r_curve, _ = precision_recall_curve(y_test, positive_proba)
                pr_auc_val = auc(r_curve, p_curve)
                log_loss_val = log_loss(y_test, y_proba)
            else:
                roc_auc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                log_loss_val = log_loss(y_test, y_proba)
        except Exception:
            pass
            
    cm = confusion_matrix(y_test, y_pred)
    
    # Business Cost Calculation (for binary classification)
    net_business_cost = None
    if problem_type == "binary" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        # Total Weighted Error Cost
        net_business_cost = float(fn * fn_cost_weight + fp * fp_cost_weight)
        
    return {
        "Accuracy": round(acc, 4),
        "Balanced Accuracy": round(bal_acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC AUC": round(roc_auc_val, 4) if roc_auc_val is not None else "N/A",
        "PR AUC": round(pr_auc_val, 4) if pr_auc_val is not None else "N/A",
        "Log Loss": round(log_loss_val, 4) if log_loss_val is not None else "N/A",
        "Business Cost Score": net_business_cost if net_business_cost is not None else "N/A",
        "Confusion Matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """
    Plots interactive Plotly confusion matrix showing counts and percentages.
    """
    cm_sum = np.sum(cm)
    cm_pct = (cm / cm_sum * 100) if cm_sum > 0 else cm
    
    labels = []
    for i in range(cm.shape[0]):
        row_labels = []
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_pct[i, j]
            row_labels.append(f"<b>{val}</b><br>({pct:.1f}%)")
        labels.append(row_labels)
        
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        text_auto=False,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        title="Interactive Confusion Matrix"
    )
    
    # Add annotations for custom formatted text
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(
                dict(
                    x=class_names[j],
                    y=class_names[i],
                    text=labels[i][j],
                    showarrow=False,
                    font=dict(size=14, color="black" if cm[i, j] < cm.max() / 2 else "white")
                )
            )
            
    fig.update_layout(annotations=annotations, template="plotly_white", title_x=0.5)
    return fig
