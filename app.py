import pandas as pd
import streamlit as st
import plotly.express as px

# Load dataset
df = pd.read_csv('5g-South Asia.csv')

# Dashboard title
st.title("Path Loss Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# **Dynamic Filter**: Seasonal Variation
seasonal_options = df["Seasonal Variation (Data Source)"].unique()
selected_seasons = st.sidebar.multiselect("Select Seasonal Variation:", seasonal_options, default=seasonal_options)

# Filter dataset based on selection
filtered_df = df[df["Seasonal Variation (Data Source)"].isin(selected_seasons)]

# Select variable to compare against Path Loss
x_var = st.sidebar.selectbox("Select a factor to compare with Path Loss:",
                            ["T-R Separation Distance (m)", "Received Power (dBm)", "Azimuth AoD (degree)",
                            "Elevation AoD (degree)", "Azimuth AoA (degree)", "Elevation AoA (degree)",
                            "RMS Delay Spread (ns)"])

# Scatter plot
fig = px.scatter(filtered_df, x=x_var, y="Path Loss (dB)", trendline="ols", title=f"Path Loss vs {x_var}")
st.plotly_chart(fig)

# Correlation heatmap
st.subheader("Correlation Matrix")
st.write("Displays how different factors correlate with Path Loss.")
correlation_matrix = filtered_df[["T-R Separation Distance (m)", "Received Power (dBm)", "Azimuth AoD (degree)",
                                  "Elevation AoD (degree)", "Azimuth AoA (degree)", "Elevation AoA (degree)",
                                  "RMS Delay Spread (ns)", "Path Loss (dB)"]].corr()
st.dataframe(correlation_matrix)
