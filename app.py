import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction")
st.caption("Indian property price estimation using Machine Learning | Currency: INR (₹)")
st.info("Academic demo: the included Hyderabad dataset is synthetic, not verified market or transaction data.")

FEATURES = ["Locality", "Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Parking"]
NUMERIC = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Parking"]

def inr(value):
    return f"₹{value:,.0f}"

def category(price):
    if price < 5_000_000: return "Budget"
    if price < 9_000_000: return "Affordable"
    if price < 15_000_000: return "Premium"
    return "Luxury"

@st.cache_data
def load_data():
    df = pd.read_csv("hyderabad_house_prices.csv")
    df["Category"] = df["Price_INR"].apply(category)
    return df

@st.cache_resource
def train_models():
    df = load_data()
    X, y = df[FEATURES], df["Price_INR"]
    prep = ColumnTransformer([
        ("locality", OneHotEncoder(handle_unknown="ignore"), ["Locality"]),
        ("numeric", StandardScaler(), NUMERIC),
    ])
    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models, rows, preds = {}, [], {}
    for name, estimator in estimators.items():
        model = Pipeline([("preprocess", prep), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        models[name], preds[name] = model, pred
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
    return models, pd.DataFrame(rows), y_test.reset_index(drop=True), preds

df = load_data()
models, results, y_test, preds = train_models()
localities = sorted(df.Locality.unique())
page = st.sidebar.radio("Navigation", ["Home", "Price Prediction", "Data Analysis", "Model Evaluation", "About"])

if page == "Home":
    st.header("Welcome")
    a, b, c = st.columns(3)
    a.metric("Dataset records", f"{len(df):,}")
    b.metric("Hyderabad localities", df.Locality.nunique())
    best = results.sort_values("R2", ascending=False).iloc[0]
    c.metric("Best model R²", f"{best.R2:.3f}")
    st.write("Predict indicative residential property prices in Hyderabad using locality and property details.")
    st.markdown("**Localities:** " + ", ".join(localities))
    st.markdown("**Workflow:** Select property details → predict price in INR → explore charts → evaluate regression models.")

elif page == "Price Prediction":
    st.header("Price Prediction")
    model_name = st.selectbox("Regression model", list(models), index=3)
    c1, c2 = st.columns(2)
    locality = c1.selectbox("Hyderabad locality", localities, index=localities.index("Uppal"))
    area = c2.number_input("Built-up area (sq. ft.)", 350, 6000, 1200, 50)
    bedrooms = c1.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5], index=1)
    bathrooms = c2.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
    age = c1.slider("Property age (years)", 0, 40, 5)
    parking = c2.selectbox("Parking spaces", [0, 1, 2, 3], index=1)
    if st.button("Predict House Price"):
        row = pd.DataFrame([{"Locality": locality, "Area_sqft": area, "Bedrooms": bedrooms,
                             "Bathrooms": bathrooms, "Age_years": age, "Parking": parking}])
        price = max(0, float(models[model_name].predict(row)[0]))
        st.success(f"Estimated property price: {inr(price)}")
        st.write(f"Category: **{category(price)}**")
        st.caption("Indicative model estimate only. Actual prices depend on project, road, amenities and market conditions.")

elif page == "Data Analysis":
    st.header("Hyderabad Housing Data Analysis")
    t1, t2, t3 = st.tabs(["Price Distribution", "Locality Comparison", "Location Map"])
    with t1:
        fig = px.histogram(df, x="Price_INR", nbins=45, title="Property Price Distribution",
                           labels={"Price_INR": "Property price (₹)"})
        st.plotly_chart(fig, use_container_width=True)
        counts = df.Category.value_counts()
        st.plotly_chart(px.pie(names=counts.index, values=counts.values, title="Price Category Share"),
                        use_container_width=True)
    with t2:
        summary = df.groupby("Locality", as_index=False).Price_INR.median().sort_values("Price_INR")
        st.plotly_chart(px.bar(summary, x="Locality", y="Price_INR", title="Median Sample Price by Locality",
                               labels={"Price_INR": "Median price (₹)"}), use_container_width=True)
    with t3:
        sample = df.sample(min(1500, len(df)), random_state=1)
        fig = px.scatter_mapbox(sample, lat="Latitude", lon="Longitude", color="Price_INR",
                                hover_name="Locality", hover_data=["Area_sqft", "Bedrooms"],
                                zoom=9, height=580, mapbox_style="open-street-map",
                                title="Sample Property Locations in Hyderabad")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    pick = st.selectbox("Select model", list(models), index=3)
    row = results[results.Model == pick].iloc[0]
    idx = np.random.RandomState(0).choice(len(y_test), min(400, len(y_test)), replace=False)
    actual, predicted = y_test.iloc[idx], preds[pick][idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers", name="Predicted vs Actual"))
    lo, hi = float(y_test.min()), float(y_test.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal prediction"))
    fig.update_layout(title="Actual vs Predicted Property Price", xaxis_title="Actual price (₹)",
                      yaxis_title="Predicted price (₹)")
    st.plotly_chart(fig, use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("MAE", inr(row.MAE))
    b.metric("RMSE", inr(row.RMSE))
    c.metric("R² Score", f"{row.R2:.3f}")
    st.subheader("Model Comparison")
    display = results.copy()
    display["MAE"] = display.MAE.map(inr)
    display["RMSE"] = display.RMSE.map(inr)
    display["R2"] = display.R2.round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)

else:
    st.header("About This Project")
    st.markdown("**Project:** Hyderabad House Price Prediction Using Regression Algorithms  \n"
                "**Region:** Hyderabad, Telangana, India  \n**Currency:** Indian Rupee (INR / ₹)  \n"
                "**Dataset:** Synthetic Hyderabad sample for academic demonstration  \n"
                "**Algorithms:** Linear Regression, Ridge Regression, Decision Tree, Random Forest  \n"
                "**Tools:** Python, pandas, scikit-learn, Plotly and Streamlit")
