import streamlit as st
import pandas as pd
import numpy as np
# import joblib
import time

# model = joblib.load("Models/diabetes_rf_model_streamlit.pkl")


st.title("Diabetes Prediction App ")

if 'name_submitted' not in st.session_state:
    st.session_state.name_submitted = False
    st.session_state.health_submitted = False


if not st.session_state.name_submitted:
    with st.form(key='name_form'):
        

        # Add input widgets inside the form
        name = st.text_input("Enter your name", key="name")
        age = st.slider("Select your age", 0, 100, 25, key="age")

        
        submitted = st.form_submit_button("Submit")

        if submitted:
            if name and age and name:  
                st.success("Form submitted successfully!")
                st.session_state.name_submitted = True
               
                st.write(f"Hello, {name}!")
                st.rerun() 
            else:
                st.warning("Please fill in all fields.")


    st.write("Outside the form")

else:
    st.write("Hello, {st.session_state.name}, 24 years old from, Ayala Bacoor (based on your IP Address). The purpose of this application is to help individuals who are at risk of diabetes. \n Kindly input all fields as accurately as possible. Please consult your physician for best diagnosis")
    st.write("We are not liable for any misdiagnosis.")
    time.sleep(3)


    if not st.session_state.name_submitted:
        with st.form(key='health_form'):
            
            st.write('Please input your height and weight:')
            height = st.number_input("Height (cm)", step=int, key="height")
            weight = st.number_input("Weight (kg)", format = "%0.1f", key="weight")

            # Add input widgets inside the form
            name = st.selectbox("Enter your name", key="name")
            age = st.slider("Select your age", 0, 100, 25, key="age")

            
            submitted = st.form_submit_button("Submit")

            if submitted:
                if name and age and name:  
                    st.success("Form submitted successfully!")
                    st.session_state.name_submitted = True
                
                    st.write(f"Hello, {name}!")
                    st.rerun() 
                else:
                    st.warning("Please fill in all fields.")



# Age = st.text_input("How old are you?")

# ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
# 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#        'HvyAlcoholConsump', 'PhysHlth', 'DiffWalk', 'Age']



# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'