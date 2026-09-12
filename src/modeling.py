import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

MODEL_CATALOG = {
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "description": "Single decision tree. Highly interpretable, fast, but prone to overfitting.",
        "default_params": {
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        }
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "description": "Ensemble of independent decision trees with bagging. Reduces variance and overfitting.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "bootstrap": True,
            "random_state": 42
        }
    },
    "Extra Trees": {
        "class": ExtraTreesClassifier,
        "description": "Extremely Randomized Trees. Fits trees with random cut-points for further variance reduction.",
        "default_params": {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "random_state": 42
        }
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "description": "Sequential boosting model. Fits each tree on the residuals of previous trees for high predictive power.",
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.8,
            "random_state": 42
        }
    },
    "XGBoost": {
        "class": XGBClassifier,
        "description": "Optimized distributed gradient boosting library. Handles complex non-linear interactions efficiently.",
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss"
        }
    },
    "LightGBM": {
        "class": LGBMClassifier,
        "description": "Fast histogram-based gradient boosting framework. Extremely quick training on large datasets.",
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.8,
            "random_state": 42,
            "verbosity": -1
        }
    }
}

def instantiate_model(model_name: str, params: Dict[str, Any]):
    """
    Instantiates model object with given hyperparameter dict.
    """
    if model_name not in MODEL_CATALOG:
        raise ValueError(f"Model '{model_name}' not supported.")
        
    model_cls = MODEL_CATALOG[model_name]["class"]
    return model_cls(**params)

def train_and_cross_validate(
    pipeline: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    n_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Runs Stratified K-Fold cross validation on Training data using a scikit-learn Pipeline.
    Strictly prevents data leakage by running preprocessing inside each fold.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Check if scoring metric is valid for problem type
    try:
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True
        )
    except Exception:
        # Fallback to accuracy if custom scoring fails
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring="accuracy", return_train_score=True
        )
        scoring = "accuracy"
        
    test_scores = cv_results['test_score']
    train_scores = cv_results['train_score']
    
    return {
        "scoring_metric": scoring,
        "mean_cv_score": float(np.mean(test_scores)),
        "std_cv_score": float(np.std(test_scores)),
        "fold_scores": [float(s) for s in test_scores],
        "mean_train_score": float(np.mean(train_scores)),
        "overfitting_gap": float(np.mean(train_scores) - np.mean(test_scores))
    }

def run_hyperparameter_tuning(
    pipeline: Pipeline,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_type: str = "GridSearch",
    n_iter: int = 10,
    n_folds: int = 3,
    scoring: str = "f1_weighted",
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Performs hyperparameter search (GridSearch or RandomizedSearch) on X_train.
    Returns best estimator pipeline and best parameters dict.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    if search_type == "GridSearch":
        search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, n_iter=n_iter, cv=cv, 
            scoring=scoring, random_state=random_state, n_jobs=-1
        )
        
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
