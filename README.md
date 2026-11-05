# Hyderabad House Price Prediction

A Streamlit ML project adapted for Hyderabad, Telangana, India. Prices are displayed in Indian Rupees (₹).

## Dataset
`hyderabad_house_prices.csv` contains 3,000 synthetic sample records across Uppal, Gachibowli, Madhapur, Kukatpally, Miyapur, LB Nagar, Kondapur, Manikonda, Kompally, Secunderabad, Nallagandla and Tellapur.

**Important:** The dataset is synthetically generated for academic demonstration. It is not an official dataset or verified record of actual property transactions. Predictions are not suitable for financial decisions.

## Features
- Price prediction by locality, area, BHK, bathrooms, property age and parking
- INR currency display, price distribution, locality comparison and map
- Linear Regression, Ridge Regression, Decision Tree and Random Forest
- MAE, RMSE and R² evaluation

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Command-line version: `python house_price_prediction.py`
