import streamlit as st
import pandas as pd
import numpy as np
# import joblib
import time
from Pages.Forms import *

# model = joblib.load("Models/diabetes_rf_model_streamlit.pkl")
st.title("Diabetes Prediction App ")


form_mapping = {
    "name_info": show_name_form,
    "bio_info" : show_bio_form,
    "lifestyle": show_lifestyle_form,
    "health_info": show_health_form,
    "wellbeing_info": show_wellbeing_form,
    "summary": show_summary,
    "results" :  show_results,  
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
    

if st.session_state.current_form != "summary" and st.session_state.current_form != "results":
    with st.form(key='main_form'):

        # 1. render UI, get validator function
        validate_fn = form_mapping[form]()  

        # 2. render submit button
        submitted = st.form_submit_button("Next ➡️")



    # 3. handle submission AFTER form
    if submitted:
        is_valid = validate_fn()  # <-- call the validator NOW
        if is_valid:
            next_form(form)
            st.rerun()
elif st.session_state.current_form == "summary":
    show_summary()
else:
    show_results()


if st.session_state.current_form != "name_info" and st.session_state.current_form != "results":
    if st.button("⬅️ Previous"):
        previous_form(st.session_state["current_form"])
        st.rerun() 

# test = st.button("Test")
# if test:
#     st.session_state.current_form = "summary"
#     st.session_state["form"] = {"name":"RAm","height":172,"weight":75,"bmi":25.351541373715524,"age":50,"gender":1,"educ":4,"fruit":"Yes","vegetable":"Yes","phys":"Yes","alcohol":"No","smoker":"No","bp":"No","chol_check":"No","chol":"No","stroke":"No","heart_disease":"No","diff_walk":"No","menthlth":1,"general_health":2,"physHlth":1,"healthcare":"No","nodoc_cost":"No","income":"Less than $10,000"}