import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import streamlit as st
import numpy as np

@st.cache_resource
def train_and_evaluate_models(df, n_splits=5):
    df = pd.get_dummies(df, columns=["Normalized Season"])

    features = ["T-R Separation Distance (m)", "Received Power (dBm)", "Azimuth AoD (degree)",
                "Elevation AoD (degree)", "Azimuth AoA (degree)", "Elevation AoA (degree)",
                "RMS Delay Spread (ns)", "Frequency (GHz)",
                "Normalized Season_fall", "Normalized Season_spring",
                "Normalized Season_summer", "Normalized Season_winter"]

    X = df[features].values
    y = df["Path Loss (dB)"].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror'),
        "Support Vector Regressor": SVR(kernel='rbf'),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    }

    results = {name: {"MAE": [], "R2": []} for name in models.keys()}

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            if "SVR" in name or "KNN" in name or "MLP" in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            results[name]["MAE"].append(mean_absolute_error(y_test, y_pred))
            results[name]["R2"].append(r2_score(y_test, y_pred))

    summary = []
    for name, scores in results.items():
        summary.append({
            "Model": name,
            "MAE (Mean)": np.mean(scores["MAE"]),
            "MAE (Std)": np.std(scores["MAE"]),
            "R2 Score (Mean)": np.mean(scores["R2"]),
            "R2 Score (Std)": np.std(scores["R2"])
        })

    return pd.DataFrame(summary)
