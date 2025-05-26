import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and data
model = pickle.load(open('LinearRegression.pkl', 'rb'))

car = pd.read_csv('Clean Car.csv')
# print(car)

# Streamlit Title
st.title("🚗 Car Price Prediction App")

# Sidebar inputs
st.sidebar.header("Enter Car Details")

# Select company
companies = sorted(car['company'].unique())
companies.insert(0, 'Select Company')
company = st.sidebar.selectbox("Select Company", companies);

# Filter car models based on company selection
if company != 'Select Company':
    models = sorted(car[car['company'] == company]['name'].unique())
else:
    models = sorted(car['name'].unique())
model_name = st.sidebar.selectbox("Select Car Model", models)

# Year selection
years = sorted(car['year'].unique(), reverse=True)
year = st.sidebar.selectbox("Select Year of Purchase", years)

# Fuel type selection
fuel_types = car['fuel_type'].unique()
fuel_type = st.sidebar.selectbox("Select Fuel Type", fuel_types)

# Kilometers driven
kms_driven = st.sidebar.number_input("Enter the Number of Kilometers the Car has Travelled", min_value=0, value=10000)

# Predict button
if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([model_name, company, year, kms_driven, fuel_type]).reshape(1, 5))
    prediction = model.predict(input_data)
    st.success(f"🚘 **Predicted Price:** ₹ {round(prediction[0], 2):,.2f}")
