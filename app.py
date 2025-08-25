import streamlit as st
import datetime
import pickle
import pandas as pd
import model as model_predict

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Flight Prediction Price")

col1, col2, col3 = st.columns([2, 1.5, 1.5])
with col1:
    airline = st.selectbox("Select airline", ["Indigo", "Air India", "Jet Airways", "SpiceJet"], index=None, placeholder="Select airline")

with col2:
    source = st.selectbox("Select source",["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"], index=None, placeholder="Select source")

with col3:
    destination = st.selectbox("Select destination",["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"], index=None, placeholder="Select destination")

col4, col5 = st.columns(2)
with col4:
    dep_time = st.time_input("Departure Time", value=None)
with col5:
    arr_time = st.time_input("Arrival Time", value=None)

col6, col7 = st.columns(2)
with col6:
    journey_date = st.date_input("Journey Date", value=None)

with col7:
    stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops"], index=None)

sample_input = pd.DataFrame([{
    'Airline': airline,
    'Source': source,
    'Destination': destination,
    'Total_Stops': stops,
    'Date': journey_date.day if journey_date else 0,
    'Month': journey_date.month if journey_date else 0,
    'Year': journey_date.year if journey_date else 0,
    'Dep_hours': dep_time.hour if dep_time else 0,
    'Dep_min': dep_time.minute if dep_time else 0,
    'Arrival_hours': arr_time.hour if arr_time else 0,
    'Arrival_min': arr_time.minute if arr_time else 0,
    'Duration_hours': arr_time.hour - dep_time.hour if arr_time and dep_time else 0,
    'Duration_min': arr_time.minute - dep_time.minute if arr_time and dep_time else 0
}])


col8, col9, col10 = st.columns([1, 2, 1])
with col9:
    if st.button("Predict", use_container_width=True):
        prediction = model.predict(sample_input)
        st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
