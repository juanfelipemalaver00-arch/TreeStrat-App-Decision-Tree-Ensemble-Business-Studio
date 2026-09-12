import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessing_pipeline, perform_stratified_split
from src.modeling import instantiate_model, train_and_cross_validate
from src.evaluation import evaluate_model_performance

def test_model_training_and_eval():
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": np.random.randn(50),
        "num2": np.random.randn(50),
        "cat1": np.random.choice(["X", "Y"], size=50),
        "target": np.random.choice([0, 1], size=50)
    })
    
    X_tr, X_te, y_tr, y_te = perform_stratified_split(df, "target", test_size=0.2, random_state=42)
    preprocessor = build_preprocessing_pipeline(X_tr, ["num1", "num2"], ["cat1"])
    
    clf = instantiate_model("Decision Tree", {"max_depth": 3})
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    
    pipe.fit(X_tr, y_tr)
    eval_res = evaluate_model_performance(pipe, X_te, y_te, problem_type="binary")
    
    assert "Accuracy" in eval_res
    assert "F1 Score" in eval_res
    assert eval_res["Accuracy"] >= 0.0
    assert eval_res["Accuracy"] <= 1.0
