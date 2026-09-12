import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

def plot_numeric_distribution(df: pd.DataFrame, col: str, target_col: Optional[str] = None):
    """
    Plots histogram distribution for a numerical feature, optionally grouped by target.
    """
    if target_col and target_col in df.columns:
        fig = px.histogram(
            df, x=col, color=target_col, barmode="overlay", marginal="rug",
            title=f"Distribution of '{col}' grouped by '{target_col}'",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    else:
        fig = px.histogram(
            df, x=col, marginal="box",
            title=f"Distribution of '{col}'",
            color_discrete_sequence=["#1f77b4"]
        )
    fig.update_layout(template="plotly_white", title_x=0.5)
    return fig

def plot_numeric_boxplot(df: pd.DataFrame, col: str, target_col: Optional[str] = None):
    """
    Plots boxplot for numeric feature to visualize quartiles and outliers.
    """
    if target_col and target_col in df.columns:
        fig = px.box(
            df, y=col, x=target_col, color=target_col,
            title=f"Boxplot of '{col}' by '{target_col}'",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    else:
        fig = px.box(
            df, y=col,
            title=f"Boxplot of '{col}'",
            color_discrete_sequence=["#2ca02c"]
        )
    fig.update_layout(template="plotly_white", title_x=0.5)
    return fig

def plot_categorical_frequency(df: pd.DataFrame, col: str, target_col: Optional[str] = None):
    """
    Plots categorical frequency bar chart, optionally stacked/grouped by target.
    """
    if target_col and target_col in df.columns:
        fig = px.histogram(
            df, x=col, color=target_col, barmode="group",
            title=f"Category Counts of '{col}' by '{target_col}'",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    else:
        counts = df[col].astype(str).value_counts().reset_index()
        counts.columns = [col, "Count"]
        fig = px.bar(
            counts, x=col, y="Count", text="Count",
            title=f"Frequency of Categories in '{col}'",
            color_discrete_sequence=["#ff7f0e"]
        )
        fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", title_x=0.5, xaxis_tickangle=-45)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Plots interactive correlation heatmap for numerical columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
        
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Pearson Correlation Heatmap"
    )
    fig.update_layout(template="plotly_white", title_x=0.5)
    return fig

def plot_target_summary(df: pd.DataFrame, target_col: str):
    """
    Plots target distribution bar chart with percentage labels.
    """
    counts = df[target_col].astype(str).value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)
    counts["Label"] = counts["Count"].astype(str) + " (" + counts["Percentage"].astype(str) + "%)"
    
    fig = px.bar(
        counts, x="Class", y="Count", text="Label", color="Class",
        title=f"Target Variable Class Distribution ('{target_col}')",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", title_x=0.5, showlegend=False)
    return fig
