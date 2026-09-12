import streamlit as st
import pandas as pd
import numpy as np
import os
import io

from src.data_ingestion import load_data, get_initial_audit
from src.data_quality import generate_quality_report, impute_or_clean_column, validate_and_cast_type
from src.eda import (
    plot_numeric_distribution, plot_numeric_boxplot, plot_categorical_frequency,
    plot_correlation_heatmap, plot_target_summary
)
from src.preprocessing import validate_target_variable, perform_stratified_split, build_preprocessing_pipeline
from src.modeling import (
    MODEL_CATALOG, instantiate_model, train_and_cross_validate, run_hyperparameter_tuning
)
from src.evaluation import evaluate_model_performance, plot_confusion_matrix
from src.interpretability import (
    get_tree_feature_importances, calculate_permutation_importance,
    plot_feature_importance_bar, get_decision_tree_rules
)
from src.business_strategy import generate_business_recommendation, generate_ai_strategy_prompt
from src.exporter import (
    export_dataframe_to_bytes, export_text_to_bytes, export_model_to_bytes, export_dict_to_json_bytes
)
from sklearn.pipeline import Pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="TreeStrat - Decision Tree & Ensemble Business Studio",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .card { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 15px; }
    .stButton>button { width: 100%; font-weight: 600; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
if 'target_info' not in st.session_state:
    st.session_state['target_info'] = None
if 'splits' not in st.session_state:
    st.session_state['splits'] = None
if 'eval_results' not in st.session_state:
    st.session_state['eval_results'] = {}
if 'trained_pipelines' not in st.session_state:
    st.session_state['trained_pipelines'] = {}

# --- Header ---
st.markdown("<div class='main-header'>🌳 TreeStrat: Decision Tree & Ensemble Business Studio</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Interactive ML Classification & Business Strategy Generator for Non-Technical Users</div>", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("📌 Workflow Steps")
steps = [
    "1. Data Upload",
    "2. Data Quality Audit",
    "3. Variable Management",
    "4. Exploratory Analysis",
    "5. Target Definition",
    "6. Train/Test Split",
    "7. Model Selection",
    "8. Hyperparameters & Tuning",
    "9. Model Evaluation",
    "10. Interpretability",
    "11. Business Recommendation",
    "12. Business Strategy & Export"
]

selected_step = st.sidebar.radio("Navigate Workflow", steps)

# Benchmark Dataset Quick Loader
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Quick Start")
if st.sidebar.button("Load Benchmark Churn Dataset"):
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_churn.csv")
    if os.path.exists(sample_path):
        st.session_state['df_raw'] = pd.read_csv(sample_path)
        st.session_state['df_clean'] = st.session_state['df_raw'].copy()
        st.sidebar.success("Benchmark dataset loaded successfully!")
    else:
        st.sidebar.error("Sample dataset file not found.")

# ==========================================
# STEP 1: DATA UPLOAD
# ==========================================
if selected_step == "1. Data Upload":
    st.header("Step 1 — Data Upload & Automatic Audit")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV, TXT, Excel .xlsx / .xls)", 
            type=["csv", "txt", "xlsx", "xls"]
        )
        delimiter = st.text_input("CSV Delimiter (default=',')", value=",")
        sheet_name = st.text_input("Excel Sheet Name or Index", value="0")
        
    with col2:
        st.info("💡 **Supported Formats**: CSV, TXT, XLSX, XLS.\nAutomatic audit will run immediately upon upload.")
        
    if uploaded_file is not None:
        try:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            sheet = int(sheet_name) if sheet_name.isdigit() else sheet_name
            df_loaded = load_data(uploaded_file, file_ext, delimiter=delimiter, sheet_name=sheet)
            st.session_state['df_raw'] = df_loaded
            st.session_state['df_clean'] = df_loaded.copy()
            st.success(f"Successfully loaded '{uploaded_file.name}' ({df_loaded.shape[0]} rows, {df_loaded.shape[1]} columns)")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            
    if st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw']
        audit = get_initial_audit(df)
        
        st.markdown("### 📊 Dataset Overview & Automatic Audit")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", audit["n_rows"])
        m2.metric("Columns", audit["n_cols"])
        m3.metric("Duplicate Rows", audit["duplicate_rows"])
        m4.metric("Total Missing Cells", audit["total_missing_cells"])
        m5.metric("Missing % Overall", f"{audit['missing_pct_overall']}%")
        
        # Diagnostic Alerts
        if audit["constant_cols"]:
            st.warning(f"⚠️ Constant Columns detected (1 unique value): `{audit['constant_cols']}`")
        if audit["high_cardinality_cols"]:
            st.warning(f"⚠️ High Cardinality Text Columns detected: `{audit['high_cardinality_cols']}`")
        if audit["text_as_numeric_cols"]:
            st.info(f"💡 Potential Numeric Columns stored as text: `{audit['text_as_numeric_cols']}`")
            
        st.markdown("### 🔍 Interactive Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# STEP 2: DATA QUALITY AUDIT
# ==========================================
elif selected_step == "2. Data Quality Audit":
    st.header("Step 2 — Data Quality Audit & Interactive Cleaning")
    
    if st.session_state['df_clean'] is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state['df_clean']
        st.markdown("### 📋 Automated Column Audit")
        
        quality_df = generate_quality_report(df)
        st.dataframe(quality_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🛠️ Interactive Column Cleaning & Missing Imputation")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            target_col_clean = st.selectbox("Select Column to Clean/Impute", options=df.columns)
        with col2:
            strategy = st.selectbox(
                "Cleaning Action", 
                options=[
                    "impute_median", "impute_mean", "impute_mode", 
                    "impute_category", "drop_rows", "drop_col"
                ],
                format_func=lambda x: {
                    "impute_median": "Impute Median (Numeric)",
                    "impute_mean": "Impute Mean (Numeric)",
                    "impute_mode": "Impute Mode (Most Frequent)",
                    "impute_category": "Create 'Missing' Category (Text)",
                    "drop_rows": "Drop Rows with Missing Values",
                    "drop_col": "Drop Entire Column"
                }[x]
            )
        with col3:
            st.write(" ")
            st.write(" ")
            if st.button("Apply Cleaning Action"):
                df_updated, msg = impute_or_clean_column(df, target_col_clean, strategy)
                st.session_state['df_clean'] = df_updated
                st.success(msg)
                st.rerun()

# ==========================================
# STEP 3: VARIABLE MANAGEMENT
# ==========================================
elif selected_step == "3. Variable Management":
    st.header("Step 3 — Variable Type Management")
    
    if st.session_state['df_clean'] is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state['df_clean']
        st.markdown("Review detected data types and override variable definitions if necessary.")
        
        var_summary = []
        for col in df.columns:
            var_summary.append({
                "Variable": col,
                "Current Type": str(df[col].dtype),
                "Unique Values": df[col].nunique(),
                "Sample Values": str(df[col].dropna().unique()[:4].tolist())
            })
        st.dataframe(pd.DataFrame(var_summary), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🔄 Explicit Data Type Override")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            col_to_cast = st.selectbox("Select Variable", options=df.columns)
        with c2:
            new_type = st.selectbox(
                "Target Type", 
                ["Numeric", "Categorical", "Integer", "Date", "String / Text"]
            )
        with c3:
            st.write(" ")
            st.write(" ")
            if st.button("Cast Variable Type"):
                df_cast, warning, success = validate_and_cast_type(df, col_to_cast, new_type)
                if success:
                    st.session_state['df_clean'] = df_cast
                    st.success(warning)
                    st.rerun()
                else:
                    st.error(warning)

# ==========================================
# STEP 4: EXPLORATORY ANALYSIS
# ==========================================
elif selected_step == "4. Exploratory Analysis":
    st.header("Step 4 — Exploratory Data Analysis (EDA)")
    
    if st.session_state['df_clean'] is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state['df_clean']
        target_col = st.session_state.get('target_col', None)
        
        t1, t2, t3 = st.tabs(["📈 Numerical Distributions", "📊 Categorical Frequencies", "🔥 Correlation Matrix"])
        
        with t1:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                col_select = st.selectbox("Select Numerical Feature", num_cols)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(plot_numeric_distribution(df, col_select, target_col), use_container_width=True)
                with c2:
                    st.plotly_chart(plot_numeric_boxplot(df, col_select, target_col), use_container_width=True)
            else:
                st.info("No numerical features found.")
                
        with t2:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                col_cat_select = st.selectbox("Select Categorical Feature", cat_cols)
                st.plotly_chart(plot_categorical_frequency(df, col_cat_select, target_col), use_container_width=True)
            else:
                st.info("No categorical features found.")
                
        with t3:
            fig_corr = plot_correlation_heatmap(df)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("At least 2 numerical features are required for a correlation matrix.")

# ==========================================
# STEP 5: TARGET DEFINITION
# ==========================================
elif selected_step == "5. Target Definition":
    st.header("Step 5 — Target Variable Selection & Problem Diagnostics")
    
    if st.session_state['df_clean'] is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state['df_clean']
        
        st.markdown("Select the target outcome variable you wish to predict.")
        target_col = st.selectbox(
            "Target Variable", 
            options=df.columns, 
            index=list(df.columns).index(st.session_state['target_col']) if st.session_state['target_col'] in df.columns else len(df.columns)-1
        )
        
        if st.button("Confirm Target Variable"):
            try:
                target_info = validate_target_variable(df, target_col)
                st.session_state['target_col'] = target_col
                st.session_state['target_info'] = target_info
                st.success(f"Target variable '{target_col}' configured as `{target_info['problem_type'].upper()}` classification!")
            except Exception as e:
                st.error(str(e))
                
        if st.session_state['target_col']:
            t_info = st.session_state['target_info']
            st.markdown("### 🎯 Target Diagnostics")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(plot_target_summary(df, st.session_state['target_col']), use_container_width=True)
            with c2:
                st.markdown(f"**Problem Type**: `{t_info['problem_type'].upper()}`")
                st.markdown(f"**Unique Categories**: `{t_info['n_unique']}`")
                st.markdown(f"**Missing Target Rows**: `{t_info['n_missing']}`")
                
                if t_info['warnings']:
                    st.markdown("#### ⚠️ Problem Warnings & Risk Checks:")
                    for w in t_info['warnings']:
                        st.warning(w)
                else:
                    st.success("✅ Target variable passes all validity checks.")

# ==========================================
# STEP 6: TRAIN / TEST SPLIT
# ==========================================
elif selected_step == "6. Train/Test Split":
    st.header("Step 6 — Stratified Train / Test Split")
    
    if not st.session_state.get('target_col'):
        st.warning("Please define a valid target variable in Step 5 first.")
    else:
        df = st.session_state['df_clean']
        target_col = st.session_state['target_col']
        
        st.markdown("Partition dataset into independent Training and Testing sets.")
        
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Test Set Percentage (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
        with c2:
            random_seed = st.number_input("Random Seed for Reproducibility", value=42, step=1)
            
        if st.button("Perform Stratified Split"):
            X_tr, X_te, y_tr, y_te = perform_stratified_split(
                df, target_col, test_size=test_size, random_state=random_seed
            )
            st.session_state['splits'] = {
                "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
                "test_size": test_size, "random_seed": random_seed
            }
            st.success(f"Split completed! Train Set: {len(X_tr)} rows, Test Set: {len(X_te)} rows.")
            
        if st.session_state['splits']:
            splits = st.session_state['splits']
            st.markdown("### 📊 Stratified Distribution Verification")
            
            df_orig_dist = df[target_col].value_counts(normalize=True).rename("Original Dataset")
            df_tr_dist = splits["y_train"].value_counts(normalize=True).rename("Training Set")
            df_te_dist = splits["y_test"].value_counts(normalize=True).rename("Test Set")
            
            dist_comp = pd.concat([df_orig_dist, df_tr_dist, df_te_dist], axis=1).round(4) * 100
            st.dataframe(dist_comp.style.format("{:.2f}%"), use_container_width=True)

# ==========================================
# STEP 7: MODEL SELECTION
# ==========================================
elif selected_step == "7. Model Selection":
    st.header("Step 7 — Model Zoo Selection")
    
    st.markdown("Select tree-based models to train and benchmark.")
    
    selected_models = []
    cols = st.columns(2)
    for idx, (m_name, m_meta) in enumerate(MODEL_CATALOG.items()):
        with cols[idx % 2]:
            st.markdown(f"#### 🌲 {m_name}")
            st.caption(m_meta["description"])
            checked = st.checkbox(f"Include {m_name} in Experiment", value=True, key=f"check_{m_name}")
            if checked:
                selected_models.append(m_name)
            st.markdown("---")
            
    st.session_state['selected_models'] = selected_models
    st.info(f"Selected **{len(selected_models)}** models for training: {', '.join(selected_models)}")

# ==========================================
# STEP 8: HYPERPARAMETERS & TUNING
# ==========================================
elif selected_step == "8. Hyperparameters & Tuning":
    st.header("Step 8 — Interactive Hyperparameter Configuration")
    
    selected_models = st.session_state.get('selected_models', ["Decision Tree", "Random Forest"])
    
    model_params = {}
    st.markdown("Configure hyperparameters for each selected model.")
    
    tabs = st.tabs(selected_models)
    for idx, m_name in enumerate(selected_models):
        with tabs[idx]:
            st.markdown(f"### Hyperparameters for `{m_name}`")
            defaults = MODEL_CATALOG[m_name]["default_params"]
            
            p_dict = {}
            if m_name == "Decision Tree":
                p_dict["max_depth"] = st.slider(f"{m_name} max_depth", 1, 20, defaults["max_depth"])
                p_dict["min_samples_split"] = st.slider(f"{m_name} min_samples_split", 2, 20, defaults["min_samples_split"])
                p_dict["criterion"] = st.selectbox(f"{m_name} criterion", ["gini", "entropy"])
            elif m_name in ["Random Forest", "Extra Trees"]:
                p_dict["n_estimators"] = st.slider(f"{m_name} n_estimators", 10, 300, defaults["n_estimators"], step=10)
                p_dict["max_depth"] = st.slider(f"{m_name} max_depth", 1, 20, defaults["max_depth"])
                p_dict["min_samples_leaf"] = st.slider(f"{m_name} min_samples_leaf", 1, 10, defaults["min_samples_leaf"])
            elif m_name in ["Gradient Boosting", "XGBoost", "LightGBM"]:
                p_dict["n_estimators"] = st.slider(f"{m_name} n_estimators", 20, 300, defaults["n_estimators"], step=10)
                p_dict["learning_rate"] = st.select_slider(f"{m_name} learning_rate", options=[0.01, 0.05, 0.1, 0.2], value=defaults["learning_rate"])
                p_dict["max_depth"] = st.slider(f"{m_name} max_depth", 1, 15, defaults["max_depth"])
                
            model_params[m_name] = p_dict
            
    st.session_state['configured_params'] = model_params

# ==========================================
# STEP 9: MODEL EVALUATION & TRAINING
# ==========================================
elif selected_step == "9. Model Evaluation":
    st.header("Step 9 & 10 — Model Training, Cross-Validation & Test Benchmarking")
    
    if not st.session_state.get('splits'):
        st.warning("Please complete Step 6 (Train/Test Split) first.")
    else:
        splits = st.session_state['splits']
        X_tr, X_te = splits["X_train"], splits["X_test"]
        y_tr, y_te = splits["y_train"], splits["y_test"]
        
        st.markdown("### ⚙️ Evaluation Settings")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_folds = st.number_input("Cross-Validation Folds", min_value=2, max_value=10, value=5)
        with c2:
            fn_cost = st.number_input("False Negative Cost Multiplier", value=3.0, step=0.5)
        with c3:
            fp_cost = st.number_input("False Positive Cost Multiplier", value=1.0, step=0.5)
            
        if st.button("🚀 Run Training & Benchmarking Pipeline"):
            selected_models = st.session_state.get('selected_models', ["Decision Tree", "Random Forest"])
            configured_params = st.session_state.get('configured_params', {})
            
            num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
            
            preprocessor = build_preprocessing_pipeline(X_tr, num_cols, cat_cols)
            
            eval_results = {}
            trained_pipelines = {}
            
            with st.spinner("Training models and running cross-validation..."):
                for m_name in selected_models:
                    params = configured_params.get(m_name, MODEL_CATALOG[m_name]["default_params"])
                    clf = instantiate_model(m_name, params)
                    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
                    
                    # 1. Run Cross Validation on Train Set
                    cv_res = train_and_cross_validate(pipeline, X_tr, y_tr, n_folds=n_folds)
                    
                    # 2. Fit on Full Train Set
                    pipeline.fit(X_tr, y_tr)
                    
                    # 3. Evaluate on Reserved Test Set
                    prob_type = st.session_state['target_info']['problem_type']
                    test_eval = evaluate_model_performance(
                        pipeline, X_te, y_te, problem_type=prob_type, 
                        fn_cost_weight=fn_cost, fp_cost_weight=fp_cost
                    )
                    
                    # Combine results
                    combined_res = {**test_eval, **cv_res}
                    eval_results[m_name] = combined_res
                    trained_pipelines[m_name] = pipeline
                    
            st.session_state['eval_results'] = eval_results
            st.session_state['trained_pipelines'] = trained_pipelines
            st.success("Training and benchmarking completed successfully!")
            
        if st.session_state['eval_results']:
            st.markdown("### 🏆 Model Comparison Leaderboard (Test Set)")
            
            comp_rows = []
            for m_name, res in st.session_state['eval_results'].items():
                comp_rows.append({
                    "Model": m_name,
                    "Accuracy": res["Accuracy"],
                    "Balanced Accuracy": res["Balanced Accuracy"],
                    "Precision": res["Precision"],
                    "Recall": res["Recall"],
                    "F1 Score": res["F1 Score"],
                    "ROC AUC": res["ROC AUC"],
                    "Mean CV Score": round(res["mean_cv_score"], 4),
                    "Overfitting Gap": round(res["overfitting_gap"], 4),
                    "Business Cost Score": res["Business Cost Score"]
                })
            df_comp = pd.DataFrame(comp_rows).sort_values(by="F1 Score", ascending=False)
            st.dataframe(df_comp, use_container_width=True)

# ==========================================
# STEP 10: INTERPRETABILITY
# ==========================================
elif selected_step == "10. Interpretability":
    st.header("Step 10 — Model Interpretability & Feature Drivers")
    
    if not st.session_state.get('trained_pipelines'):
        st.warning("Please run Step 9 (Model Training) first.")
    else:
        pipelines = st.session_state['trained_pipelines']
        selected_m = st.selectbox("Select Model to Interpret", list(pipelines.keys()))
        pipeline = pipelines[selected_m]
        
        splits = st.session_state['splits']
        X_te, y_te = splits["X_test"], splits["y_test"]
        
        t1, t2 = st.tabs(["🌳 Feature Importances (Gini/MDI)", "📜 Decision Tree Rules"])
        
        with t1:
            try:
                num_cols = splits["X_train"].select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = splits["X_train"].select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Retrieve feature names
                df_imp = get_tree_feature_importances(pipeline, X_te.columns.tolist())
                if not df_imp.empty:
                    st.plotly_chart(plot_feature_importance_bar(df_imp, title=f"Top Driver Features — {selected_m}"), use_container_width=True)
                else:
                    st.info("Feature importance not directly supported for this model structure.")
            except Exception as e:
                st.error(f"Error computing feature importances: {str(e)}")
                
        with t2:
            rules_text = get_decision_tree_rules(pipeline, X_te.columns.tolist())
            st.code(rules_text, language="text")

# ==========================================
# STEP 11: BUSINESS RECOMMENDATION
# ==========================================
elif selected_step == "11. Business Recommendation":
    st.header("Step 11 — Business-Oriented Model Recommendation")
    
    if not st.session_state.get('eval_results'):
        st.warning("Please run Step 9 (Model Training) first.")
    else:
        results = st.session_state['eval_results']
        primary_metric = st.selectbox("Primary Optimization Metric", ["F1 Score", "Recall", "Precision", "Accuracy", "ROC AUC"])
        
        rec = generate_business_recommendation(results, primary_metric=primary_metric)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### 🏅 Recommended Model: `{rec['recommended_model']}`")
        st.markdown(rec['justification'])
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# STEP 12: BUSINESS STRATEGY & EXPORT
# ==========================================
elif selected_step == "12. Business Strategy & Export":
    st.header("Step 12 — Business Strategy AI Prompt & Export Center")
    
    if not st.session_state.get('eval_results'):
        st.warning("Please run Step 9 (Model Training) first.")
    else:
        results = st.session_state['eval_results']
        rec = generate_business_recommendation(results)
        win_m = rec['recommended_model']
        win_res = results[win_m]
        
        st.markdown("### 🤖 Generative AI Business Strategy Prompt")
        st.caption("Copy this prompt into ChatGPT, Gemini, or Claude to auto-generate an executive strategy roadmap.")
        
        problem_desc = st.text_input("Enter Business Context Description", value="Customer Churn Prevention & Retention Optimization")
        
        ai_prompt = generate_ai_strategy_prompt(
            problem_description=problem_desc,
            target_var=st.session_state['target_col'],
            recommended_model=win_m,
            metrics_summary=win_res,
            top_features=list(st.session_state['df_clean'].columns),
            dataset_info=get_initial_audit(st.session_state['df_clean'])
        )
        
        st.text_area("Generated Prompt", value=ai_prompt, height=350)
        
        st.markdown("---")
        st.markdown("### 📥 Export Center")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Download Cleaned Dataset (CSV)",
                data=export_dataframe_to_bytes(st.session_state['df_clean'], "csv"),
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )
        with c2:
            st.download_button(
                "Download AI Prompt (.md)",
                data=export_text_to_bytes(ai_prompt),
                file_name="ai_business_strategy_prompt.md",
                mime="text/markdown"
            )
        with c3:
            if win_m in st.session_state.get('trained_pipelines', {}):
                model_bytes = export_model_to_bytes(st.session_state['trained_pipelines'][win_m])
                st.download_button(
                    f"Download Trained `{win_m}` Model (.pkl)",
                    data=model_bytes,
                    file_name=f"{win_m.lower().replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream"
                )
