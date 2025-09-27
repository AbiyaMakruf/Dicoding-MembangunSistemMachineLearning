import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

train_path = "insurance_preprocessing/insurance_train_preprocessed.csv"
test_path = "insurance_preprocessing/insurance_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# Lokal
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# DagsHub
dagshub.init(repo_owner='abiyamf', repo_name='my-first-repo', mlflow=True)

mlflow.set_experiment("Insurance_Cost_Prediction_Tuning")

n_estimators_range = np.linspace(50, 500, 5, dtype=int)   # 5 nilai antara 50-500
max_depth_range = np.linspace(5, 50, 5, dtype=int)        # 5 nilai antara 5-50

best_score = -np.inf
best_params = {}
best_model = None

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"rf_{n_estimators}_{max_depth}"):
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # Manual logging
            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("MAE", mae)    # tambahan 1
            mlflow.log_metric("MAPE", mape)  # tambahan 2

            # Simpan model terbaik
            if r2 > best_score:
                best_score = r2
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                best_model = model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="best_model",
                    input_example=X_test.iloc[:1]
                )

print("Best R2:", best_score)
print("Best Params:", best_params)
