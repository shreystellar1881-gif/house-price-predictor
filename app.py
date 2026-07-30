import streamlit as st
import joblib
import pandas as pd

# Load the trained model and feature names
model = joblib.load("random_forest_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Page setup
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")
st.title("🏡 California House Price Predictor")
st.write("Enter the property details below to estimate the median house value.")

# --- Location inputs ---
st.subheader("📍 Location")
col1, col2 = st.columns(2)
with col1:
    latitude = st.slider("Latitude", 32.5, 42.0, 34.0)
with col2:
    longitude = st.slider("Longitude", -124.5, -114.0, -118.0)

# --- Property details ---
st.subheader("🏠 Property Details")
col3, col4 = st.columns(2)
with col3:
    house_age = st.slider("House Age (years)", 1, 52, 20)
    avg_rooms = st.slider("Average Rooms", min_value=2, max_value=10, value=5, step=1)
with col4:
    avg_bedrooms = st.slider("Average Bedrooms", min_value=1, max_value=5, value=2, step=1)
    avg_occup = st.slider("Average Occupancy", 0.5, 10.0, 3.0)

# --- Economic / area factors ---
st.subheader("💰 Economic Factors")
col5, col6 = st.columns(2)
with col5:
    med_inc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5)
with col6:
    population = st.slider("Population", 3, 35000, 1000)

st.divider()

# --- Predict button ---
if st.button("Predict Price", use_container_width=True):
    input_dict = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": avg_rooms,
        "AveBedrms": avg_bedrooms,
        "Population": population,
        "AveOccup": avg_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    input_df = pd.DataFrame([input_dict])[feature_names]
    prediction = model.predict(input_df)[0]
    price_in_dollars = prediction * 100000

    st.success(f"### Estimated House Price: ${price_in_dollars:,.2f}")

    # --- Context: compare to dataset median ---
    # California Housing dataset median house value (in $100k units) ≈ 1.797
    dataset_median = 1.797 * 100000
    diff = price_in_dollars - dataset_median
    if diff > 0:
      st.info(f"This is **\\${diff:,.2f} above** the dataset's median house price (\\${dataset_median:,.2f}).")
    else:
        st.info(f"This is **\\${abs(diff):,.2f} below** the dataset's median house price (\\${dataset_median:,.2f}).")
    

        # --- Location map ---
    st.subheader("🗺️ Property Location")
    map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_df, zoom=9)

    
    # --- Feature importance chart ---
    st.subheader("📊 What drove this prediction?")
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    st.bar_chart(importance_df.set_index("Feature"))

