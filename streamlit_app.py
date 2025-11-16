import streamlit as st
import pandas as pd
import numpy as np
# import joblib
import time
from Pages.Forms import show_name_form, show_bio_form, show_lifestyle_form, show_health_form, show_summary, next_form

# model = joblib.load("Models/diabetes_rf_model_streamlit.pkl")
st.title("Diabetes Prediction App ")


form_mapping = {
    "name_info": show_name_form,
    "bio_info" : show_bio_form,
    "lifestyle": show_lifestyle_form,
    "health_info": show_health_form,
    "summary": show_summary    
}


if "form" not in st.session_state:
    st.session_state["form"] = {}
if "age" not in st.session_state:
    st.session_state["age"] = 0
if "gender" not in st.session_state:
    st.session_state["gender"] = ""



if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

if 'current_form' not in st.session_state:
    st.session_state.current_form = "name_info"


form = st.session_state["current_form"]

message= """This Diabetes Risk Assessment App was developed using a machine learning model trained on health and lifestyle data from U.S. adults in the 2021 National Health Interview Survey. Its purpose is to give individuals a clearer sense of their potential risk for diabetes based on their personal information, habits, and health indicators.

Please note that this tool provides an estimate only and is not a medical diagnosis. For proper evaluation, interpretation, and medical advice, it is always best to consult a licensed healthcare professional."""
st.write(message)
    

if st.session_state.current_form != "summary":
    with st.form(key='main_form'):

        # 1. render UI, get validator function
        validate_fn = form_mapping[form]()  

        # 2. render submit button
        submitted = st.form_submit_button("Next")

    # 3. handle submission AFTER form
    if submitted:
        is_valid = validate_fn()  # <-- call the validator NOW
        if is_valid:
            next_form(form)
            st.rerun()
else:
    show_summary()