import streamlit as st
import plotly.express as px
import numpy as np

from ml import train_and_evaluate_models

def displayScatterPlot(filtered_df):
    st.subheader("Path Loss Scatter Plot")

    trendline_type = st.selectbox("Select trendline type:", ["ols", "lowess", None])

    x_var = st.selectbox("Select a factor to compare with Path Loss:",
                        ["T-R Separation Distance (m)", "Received Power (dBm)", "Azimuth AoD (degree)",
                         "Elevation AoD (degree)", "Azimuth AoA (degree)", "Elevation AoA (degree)",
                         "RMS Delay Spread (ns)"])

    scatter_fig = px.scatter(filtered_df, x=x_var, y="Path Loss (dB)",
                           trendline=trendline_type,
                           title=f"Path Loss vs {x_var}",
                           trendline_color_override="red")
    st.plotly_chart(scatter_fig)

def displayBoxPlot(filtered_df):
    st.subheader("Seasonal Impact on Path Loss")
    season_fig = px.box(filtered_df, x="Normalized Season", y="Path Loss (dB)",
                       color="Normalized Season",
                       title="Path Loss Distribution Across Seasons")
    st.plotly_chart(season_fig)

def displayCorrelationHeatmap(filtered_df):
    st.subheader("Correlation Heatmap")

    # Compute correlation matrix
    numeric_df = filtered_df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().round(2)

    fig = px.imshow(corr_matrix,
                   text_auto=True,
                   color_continuous_scale='RdBu_r',
                   title="Correlation Matrix of Numerical Features",
                   aspect='auto')

    st.plotly_chart(fig)

def displayMachineLearningCharts(df):
    st.header("Machine Learning Model Evaluation")

    results_df = train_and_evaluate_models(df, n_splits=5)
    best_model = results_df.loc[results_df["R2 Score (Mean)"].idxmax()]

    st.success(f"🏆 **Best Model:** {best_model['Model']}")
    st.write(f"🔹 R² Score: {best_model['R2 Score (Mean)']:.4f}")
    st.write(f"🔹 MAE: {best_model['MAE (Mean)']:.4f}")

    sorted_mae_df = results_df.sort_values(by="MAE (Mean)", ascending=True)
    sorted_r2_df = results_df.sort_values(by="R2 Score (Mean)", ascending=False)

    st.plotly_chart(
        px.bar(
            sorted_mae_df,
            x="Model",
            y="MAE (Mean)",
            title="Mean Absolute Error (MAE) by Model (K-Fold Avg)",
            color="MAE (Mean)",
            color_continuous_scale="blues"
        ).update_traces(hovertemplate='Model: %{x}<br>MAE: %{y:.2f}')
    )

    st.plotly_chart(
        px.bar(
            sorted_r2_df,
            x="Model",
            y="R2 Score (Mean)",
            title="R² Score by Model (K-Fold Avg)",
            color="R2 Score (Mean)",
            color_continuous_scale="viridis"
        ).update_traces(hovertemplate='Model: %{x}<br>R² Score: %{y:.2f}')
    )
