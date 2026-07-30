# 🏡 California House Price Predictor

An end-to-end machine learning project that predicts median house prices in California and serves the model through an interactive Streamlit dashboard.
**📓 Google Colab Notebook :** (https://colab.research.google.com/drive/1a-xqcWjOR4y5oyrg1qeuZyASNp3DojYA?usp=sharing)

**🔗 Live App:** [house-price-predictor-5lz5m2xfwbkfyaqpdyocaf.streamlit.app](https://house-price-predictor-5lz5m2xfwbkfyaqpdyocaf.streamlit.app/)

---

## Overview

This project predicts median house values using the California Housing dataset, comparing multiple regression models before selecting the best performer for deployment. The final model is wrapped in a Streamlit dashboard that lets users adjust property and location features via sliders and get a live price estimate, along with supporting context (feature importance, comparison to the dataset median, and a location map).

## Problem

Predict the median house value for a district in California based on features like median income, house age, average rooms/bedrooms, population, occupancy, and location (latitude/longitude).

## Dataset

- **Source:** California Housing dataset (`sklearn.datasets.fetch_california_housing`)
- **Size:** ~20,000 records
- **Features:** Median Income, House Age, Average Rooms, Average Bedrooms, Population, Average Occupancy, Latitude, Longitude
- **Target:** Median house value (in $100,000s)

## Approach

1. **EDA & preprocessing** — explored feature distributions, checked correlations with target, handled scaling as needed
2. **Model comparison** — trained and evaluated three models on a held-out test set:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.533 | 0.746 | 0.576 |
| **Random Forest** | **0.328** | **0.505** | **0.805** |
| Gradient Boosting | 0.372 | 0.542 | 0.776 |

3. **Model selection** — Random Forest performed best across all three metrics and was selected for deployment
4. **Deployment** — model serialized with `joblib`, served through a Streamlit app, hosted on Streamlit Community Cloud

## Dashboard Features

- Interactive sliders for all input features, grouped by category (location, property details, economic factors)
- Live price prediction on demand
- Comparison of predicted price against the dataset's median house price
- Feature importance chart showing which inputs most influenced the prediction
- Location map pinpointing the selected latitude/longitude

## Tech Stack

- **Modeling:** Python, scikit-learn, pandas, NumPy (developed in Google Colab)
- **App/Deployment:** Streamlit, joblib, GitHub, Streamlit Community Cloud

## What I Learned

- Regression modeling and honest model comparison using MAE/RMSE/R²
- Feature engineering and evaluation methodology
- Building and deploying an interactive ML-powered web app
- Debugging real deployment issues (large file handling for GitHub, environment setup, git workflow)

## Run Locally

```bash
git clone https://github.com/shreystellar1881-gif/house-price-predictor.git
cd house-price-predictor
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
house-price-predictor/
├── app.py                     # Streamlit app
├── random_forest_model.pkl    # Trained model
├── feature_names.pkl          # Feature order used during training
├── requirements.txt           # Dependencies
└── README.md
```
