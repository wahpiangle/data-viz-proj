import pandas as pd
import streamlit as st
from components import displayBoxPlot, displayScatterPlot, displayCorrelationHeatmap, displayMachineLearningCharts

df = pd.read_csv('5g-South Asia.csv')
df["Normalized Season"] = df["Seasonal Variation (Data Source)"].str.lower().str.extract(r'(fall|spring|summer|winter)')

st.title("Path Loss Analysis Dashboard")
st.sidebar.header("Filters")

# Seasonal filter
seasonal_options = df["Normalized Season"].dropna().unique()
selected_seasons = st.sidebar.multiselect("Select Seasonal Variation:",
                                         seasonal_options,
                                         default=seasonal_options)

filtered_df = df[df["Normalized Season"].isin(selected_seasons)]

# Distance filter
min_dist, max_dist = st.sidebar.slider(
    "Filter by T-R Separation Distance (m)",
    int(df["T-R Separation Distance (m)"].min()),
    int(df["T-R Separation Distance (m)"].max()),
    (int(df["T-R Separation Distance (m)"].min()), int(df["T-R Separation Distance (m)"].max()))
)

filtered_df = filtered_df[
    (filtered_df["T-R Separation Distance (m)"] >= min_dist) &
    (filtered_df["T-R Separation Distance (m)"] <= max_dist)
]

# Chart selection options
st.sidebar.header("Chart Selection")
show_scatter_plot = st.sidebar.checkbox("Show Scatter Plot", value=True)
show_seasonal_boxplot = st.sidebar.checkbox("Show Seasonal Impact Box Plot", value=True)
show_correlation_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
show_machine_learning = st.sidebar.checkbox("Show Machine Learning Model Evaluation", value=True)

if show_scatter_plot:
    displayScatterPlot(filtered_df)

# Seasonal boxplot
if show_seasonal_boxplot:
    displayBoxPlot(filtered_df)

# Correlation heatmap
if show_correlation_heatmap:
    displayCorrelationHeatmap(filtered_df)

if show_machine_learning:
    displayMachineLearningCharts(df)
