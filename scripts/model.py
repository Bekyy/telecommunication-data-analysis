import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Start MLflow run
with mlflow.start_run():

    # Log the model version and parameters
    mlflow.log_param("model_version", "1.0")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Load and split your data
    features = ['number_of_xdr_sessions', 'total_session_duration', 'total_traffic',
                'Avg Bearer TP UL (kbps)', 'Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (MB)', 
                'TCP UL Retrans. Vol (MB)', 'Avg RTT UL (sec)', 'Avg RTT DL (sec)']
    X = merged_data[features]
    y = merged_data['Satisfaction Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Log metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Log any artifact (like CSVs or plots)
    metrics_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
    metrics_df.to_csv("/tmp/metrics.csv", index=False)
    mlflow.log_artifact("/tmp/metrics.csv")
