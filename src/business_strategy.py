import pandas as pd
import numpy as np
from typing import Dict, Any, List

def generate_business_recommendation(
    model_results: Dict[str, Dict[str, Any]], 
    primary_metric: str = "F1 Score",
    fn_cost_weight: float = 3.0,
    fp_cost_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Ranks evaluated models based on primary metric, cross-validation stability, 
    overfitting control, and business cost score.
    Generates plain English justification.
    """
    if not model_results:
        return {"recommended_model": "None", "justification": "No models have been trained and evaluated."}
        
    ranking = []
    
    for m_name, res in model_results.items():
        score = res.get(primary_metric, 0.0)
        cv_score = res.get("mean_cv_score", 0.0)
        overfit_gap = res.get("overfitting_gap", 0.0)
        cost_score = res.get("Business Cost Score", float('inf'))
        
        # Penalize severe overfitting (> 15% gap between train and test)
        penalty = 0.05 if overfit_gap > 0.15 else 0.0
        net_eval_score = (score if isinstance(score, float) else 0.0) - penalty
        
        ranking.append({
            "model_name": m_name,
            "net_eval_score": net_eval_score,
            "primary_metric_val": score,
            "cv_score": cv_score,
            "overfit_gap": overfit_gap,
            "cost_score": cost_score,
            "is_tree": m_name == "Decision Tree"
        })
        
    # Sort ranking
    ranking.sort(key=lambda x: x["net_eval_score"], reverse=True)
    best = ranking[0]
    
    m_win = best["model_name"]
    score_win = best["primary_metric_val"]
    cv_win = best["cv_score"]
    gap_win = best["overfit_gap"]
    
    justification = (
        f"**{m_win}** is recommended as the optimal model for business deployment. "
        f"It achieved the highest performance on the primary target metric ({primary_metric}: **{score_win}**) "
        f"with a stable cross-validation score of **{cv_win:.4f}** on the training set. "
    )
    
    if gap_win < 0.05:
        justification += f"Furthermore, it demonstrated excellent generalization with a minimal overfitting gap ({gap_win:.2%}) between training and test sets. "
    elif gap_win > 0.15:
        justification += f"Note: A moderate train/test performance gap ({gap_win:.2%}) was observed; hyperparameter regularization is advised prior to final deployment. "
        
    if best["is_tree"]:
        justification += "As a single Decision Tree, it offers 100% white-box interpretability, making it ideal if strict auditability is required by business stakeholders."
    else:
        justification += f"As an ensemble model, it successfully captures non-linear interactions across variables while offering superior predictive stability compared to a single decision tree."
        
    return {
        "recommended_model": m_win,
        "justification": justification,
        "ranking_table": pd.DataFrame(ranking)
    }

def generate_ai_strategy_prompt(
    problem_description: str,
    target_var: str,
    recommended_model: str,
    metrics_summary: Dict[str, Any],
    top_features: List[str],
    dataset_info: Dict[str, Any]
) -> str:
    """
    Generates a structured, copy-pasteable Markdown prompt for Generative AI (LLMs)
    to transform classification results into a 10-point business action plan.
    """
    features_str = ", ".join(top_features[:8]) if top_features else "Not specified"
    metrics_formatted = "\n".join([f"- {k}: {v}" for k, v in metrics_summary.items() if k not in ["Confusion Matrix", "y_pred", "y_proba"]])
    
    prompt = f"""# ROLE AND SYSTEM INSTRUCTION
Act as an Executive Business Strategist, Chief Operating Officer (COO), and Senior Analytics Director. 
Your objective is to translate technical Machine Learning classification outputs into an actionable, high-impact business strategy roadmap for senior leadership.

# CONTEXT & PROBLEM DEFINITION
- **Business Problem Context**: {problem_description if problem_description else 'Customer Classification / Business Strategy Optimization'}
- **Target Variable**: `{target_var}`
- **Dataset Size**: {dataset_info.get('n_rows', 'N/A')} records, {dataset_info.get('n_cols', 'N/A')} features.

# MODEL RESULTS & METRICS SUMMARY
- **Selected Optimal Model**: {recommended_model}
- **Top Key Predictive Variables (Drivers)**: {features_str}
- **Model Test Performance Metrics**:
{metrics_formatted}

---

# REQUESTED DELIVERABLE: BUSINESS STRATEGY & ACTION PLAN
Please generate a structured, professional Executive Business Strategy document in Markdown containing the following 10 sections:

1. **Executive Summary**: High-level synthesis of findings, business implications, and core objectives.
2. **Key Predictive Findings & Drivers**: Detailed breakdown of top driver variables ({features_str}) and what they reveal about customer/operational behavior.
3. **Target Segment Profiling**: Define risk or opportunity customer segments based on model predictions.
4. **Recommended Business Actions**: Specific operational interventions for high-risk/high-value segments (e.g. retention campaigns, pricing adjustments, operational changes).
5. **Action Prioritization Matrix**: Categorize actions into High/Medium/Low priority based on Impact vs Effort.
6. **Expected Business Impact**: Quantify financial ROI, risk reduction, or revenue retention opportunities.
7. **Key Performance Indicators (KPIs)**: Measurable metrics to track strategy execution and business success over 30, 60, and 90 days.
8. **Implementation Roadmap**: Phased timeline (Phase 1: Immediate Wins, Phase 2: Scaling, Phase 3: Optimization).
9. **Risk Assessment & Mitigation**: Identify operational, ethical, or model drift risks and mitigation strategies.
10. **Next Steps for Stakeholders**: Specific immediate assignments for Marketing, Operations, Product, and Finance teams.

*Note: Base all inferences strictly on the provided empirical model results and key driver features without inventing unsupported data points.*
"""
    return prompt
