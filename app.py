import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_csv('5g-South Asia.csv')
st.title("Path Loss Analysis Dashboard")
st.sidebar.header("Filters")

df["Normalized Season"] = df["Seasonal Variation (Data Source)"].str.lower().str.extract(r'(fall|spring|summer|winter)')
seasonal_options = df["Normalized Season"].unique()
selected_seasons = st.sidebar.multiselect("Select Seasonal Variation:", seasonal_options, default=seasonal_options)

filtered_df = df[df["Normalized Season"].isin(selected_seasons)]

x_var = st.sidebar.selectbox("Select a factor to compare with Path Loss:",
                            ["T-R Separation Distance (m)", "Received Power (dBm)", "Azimuth AoD (degree)",
                            "Elevation AoD (degree)", "Azimuth AoA (degree)", "Elevation AoA (degree)",
                            "RMS Delay Spread (ns)"])

fig = px.scatter(filtered_df, x=x_var, y="Path Loss (dB)", trendline="ols", title=f"Path Loss vs {x_var}", trendline_color_override="red")
st.plotly_chart(fig)

st.subheader("Correlation Matrix")
st.write("Displays how different factors correlate with Path Loss.")
correlation_matrix = px.imshow(filtered_df[["T-R Separation Distance (m)", "Received Power (dBm)", "Azimuth AoD (degree)",
                                    "Elevation AoD (degree)", "Azimuth AoA (degree)", "Elevation AoA (degree)",
                                    "RMS Delay Spread (ns)", "Path Loss (dB)"]].corr(), width=800, height=800)
st.plotly_chart(correlation_matrix, use_container_width=True)
