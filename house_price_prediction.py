import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("hyderabad_house_prices.csv")
features = ["Locality", "Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Parking"]
numeric = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Parking"]
X, y = df[features], df["Price_INR"]
prep = ColumnTransformer([("locality", OneHotEncoder(handle_unknown="ignore"), ["Locality"]),
                          ("numeric", StandardScaler(), numeric)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
estimators = {"Linear Regression": LinearRegression(), "Ridge Regression": Ridge(alpha=1.0),
              "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
              "Random Forest": RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1)}
results = []
trained = {}
for name, estimator in estimators.items():
    model = Pipeline([("preprocess", prep), ("model", estimator)])
    model.fit(X_train, y_train)
    trained[name] = model
    pred = model.predict(X_test)
    results.append({"Model": name, "MAE (INR)": mean_absolute_error(y_test, pred),
                    "RMSE (INR)": np.sqrt(mean_squared_error(y_test, pred)), "R2 Score": r2_score(y_test, pred)})
result_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
print("First five Hyderabad dataset records:\\n", df.head())
print("\\nModel comparison (INR):\\n", result_df.to_string(index=False))
best = result_df.iloc[0]["Model"]
pred = trained[best].predict(X_test)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred, alpha=0.35)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual price (₹)")
plt.ylabel("Predicted price (₹)")
plt.title(f"Hyderabad Actual vs Predicted ({best})")
plt.tight_layout()
plt.savefig("actual_vs_predicted_hyderabad.png")
plt.show()
