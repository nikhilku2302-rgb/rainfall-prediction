import streamlit as st
import pandas as pd
import joblib 


# ==============================
# Load saved model and encoding
# ==============================
model = joblib.load("rainfall_prediction_model.pkl")
encoded_columns = joblib.load("encoded_columns.pkl")
print(model)


st.set_page_config(page_title="Rainfall Predictor", layout="centered")

st.title("🌧️ Rainfall Prediction App")
st.write("Enter features to predict the Rainfall")

# ==============================
# User Inputs
# ==============================
pressure = st.number_input("pressure ", min_value=100, max_value=10000, step=100)
maxtemp = st.number_input("maxtemp", min_value=1, max_value=1000, step=1)
temperature = st.number_input("temperature", min_value=1, max_value=1000, step=1)
mintemp = st.number_input("mintemp", min_value=1, max_value=1000, step=1)
dewpoint = st.number_input("dewpoint", min_value=0, max_value=500, step=1)
humidity = st.number_input("humidity", min_value=10, max_value=500, step=1)
cloud = st.number_input("cloud", min_value=0, max_value=100, step=1)
sunshine = st.number_input("sunshine", min_value=1, max_value=10, step=1)
winddirection = st.number_input("winddirection", min_value=1, max_value=100, step=1)
windspeed = st.number_input("windspeed", min_value=1, max_value=150, step=1)


# humidity = st.selectbox("Main Road Access", ["yes", "no"])


city = st.text_input("City", "Noida")

# ==============================
# Predict Button
# ==============================
if st.button("Predict "):
    # Create raw input dataframe
    input_data = pd.DataFrame([{
        'pressure': pressure,
        'maxtemp': maxtemp,
        'temperature': temperature,
        'mintemp': mintemp,
        'dewpoint': dewpoint,
        'humidity': humidity,
        'cloud': cloud,
        'sunshine': sunshine,
        'winddirection': winddirection,
        'windspeed': windspeed,
        
        'city': city
    }])

    # Encode input
    input_encoded = pd.get_dummies(input_data)

    # Align columns with training data
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # Prediction
    prediction = model['model0'].predict(input_encoded)[0]
    prediction_result = "Rainfall" if prediction == 1 else "No Rainfall"
    # prediction = model.dict(input_encoded)[0]
    st.success(f"🌦️ Predicted rainfall:  {prediction_result}")

