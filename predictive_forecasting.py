"""
Power BI Internship — Week 3
Predictive Analytics Using Historical Data Script
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def generate_historical_data(months=36):
    """Generates 36 months of historical monthly revenue data."""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    
    dates = []
    revenue = []
    
    base_revenue = 45000.0
    for i in range(months):
        current_date = start_date + timedelta(days=i*30.5)
        trend = i * 1800.0
        seasonality = np.sin(i / 2.0) * 4000.0
        noise = np.random.normal(0, 1500.0)
        
        monthly_val = round(base_revenue + trend + seasonality + noise, 2)
        dates.append(current_date.strftime("%Y-%m-01"))
        revenue.append(monthly_val)
        
    df = pd.DataFrame({"Date": dates, "Revenue": revenue})
    return df

def run_predictive_forecasting():
    print("[*] Loading and preprocessing historical dataset...")
    df = generate_historical_data(36)
    
    # Preprocessing & Feature Engineering
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month_Index"] = np.arange(len(df))
    
    # Train / Test Split
    X = df[["Month_Index"]]
    y = df["Revenue"]
    
    print("[*] Training Linear Regression Forecasting Model...")
    model = LinearRegression()
    model.fit(X, y)
    
    # Model Evaluation Metrics
    y_pred_hist = model.predict(X)
    r2 = r2_score(y, y_pred_hist)
    rmse = np.sqrt(mean_squared_error(y, y_pred_hist))
    
    print(f"[✔] Model Evaluation -> R^2 Score: {r2:.4f} | RMSE: ${rmse:.2f}")
    
    df["Data_Type"] = "Historical"
    df["Predicted_Revenue"] = np.round(y_pred_hist, 2)
    
    # Forecast Future 12 Months
    print("[*] Generating 12-Month Future Predictions...")
    future_indices = np.arange(len(df), len(df) + 12)
    future_dates = [df["Date"].iloc[-1] + pd.DateOffset(months=i) for i in range(1, 13)]
    future_preds = model.predict(future_indices.reshape(-1, 1))
    
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Revenue": np.nan,
        "Month_Index": future_indices,
        "Data_Type": "Forecasted",
        "Predicted_Revenue": np.round(future_preds, 2)
    })
    
    final_df = pd.concat([df, future_df], ignore_index=True)
    
    # Export clean datasets
    final_df.to_csv("historical_and_forecasted_sales.csv", index=False)
    final_df.to_excel("historical_and_forecasted_sales.xlsx", index=False)
    
    print("[✔] Forecasting Complete! Exported 'historical_and_forecasted_sales.csv'.")

if __name__ == "__main__":
    run_predictive_forecasting()