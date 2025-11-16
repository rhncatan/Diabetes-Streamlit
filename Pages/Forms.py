import streamlit as st
import time
import numpy as np


gender_mapping = {"Male":1,
                  "Female":0}
bool_mapping = {"Yes":1,
                "No":0}
gender_options = ["-- Select Gender --"] + list(gender_mapping.keys())
PLACEHOLDER = "-- Select Answer --"
bool_options = [PLACEHOLDER] + list(bool_mapping.keys())


def next_form(session_key):
    keys = ['name_info','bio_info','lifestyle','health_info','summary']

    idx = keys.index(session_key)

    if idx == len(keys) - 1:  # last page
        st.session_state["current_form"] = "summary"
    else:
        st.session_state["current_form"] = keys[idx + 1]


def show_name_form():
    name = st.text_input("Enter your name")

    def validate():
        if not name:
            st.warning("Please fill in all fields.")
            return False
        
        st.session_state["form"]["name"] = name
        return True
    return validate        


def show_bio_form():
    st.header("Bio Information:")
    st.write(f"How old are you, {st.session_state['form']['name']}?")

    age = st.slider("Select your age", 0, 100, 25)

    st.write(f"What is your assigned sex at birth?")

    gender = st.selectbox("Gender:", options=gender_options)
    st.write('Please input your height and weight:')

    height = st.number_input(
        "Height (cm)",
        step=1,
        value=1,
        min_value=1
    )

    weight = st.number_input(
        "Weight (kg)",
        format="%0.1f",
        value=1.0,
        min_value=1.0
    )
    bmi = weight / ((height / 100) ** 2)

    def validate():
        if height <= 0 or weight <= 0 or gender is None:
            st.warning("Please fill in all fields.")
            return False
        
        if gender == "-- Select Gender --":
            st.warning("Please choose a gender")
            return False
        st.session_state["form"]["height"] = height
        st.session_state["form"]["weight"] = weight
        st.session_state["form"]["bmi"] = bmi
        st.session_state["form"]["age"] = age
        st.session_state["form"]["gender"] = gender_mapping[gender]
        return True
    
    return validate

def show_lifestyle_form():
    st.header("Lifestyle Information: ")
    fruit = st.selectbox("Do you eat any fruit/s at least once (1) per day?",                          
                         options=bool_options)
    
    vegetable = st.selectbox("Do you eat any vegetable/s at least once (1) per day?",
                             options=bool_options)
    
    phys = st.selectbox("Have you done any physical activity in the past 30 days (not including job)?",
                       options=bool_options)
    
    gender_value = st.session_state.form.get("gender", None)
    if gender_value == 1:
        num_glasses = 14
    else:
        num_glasses = 7  
    alcohol = st.selectbox(f"Do you drink more than (>=) {num_glasses} glasses of alcohol per week?",
                           options=bool_options)

    smoker = st.selectbox(f"Have you smoked at least 100 cigarettes in your entire life?",
                           options=bool_options,
                           help="Note: 5 packs = 100 cigarettes")
    def validate():
        answers = [fruit, vegetable, phys, alcohol, smoker]
        if any(ans == PLACEHOLDER for ans in answers):
            st.warning("Please select a valid answer for all questions.")
            return False
        
        st.session_state["form"]["fruit"] = fruit
        st.session_state["form"]["vegetable"] = vegetable
        st.session_state["form"]["phys"] = phys
        st.session_state["form"]["alcohol"] = alcohol
        st.session_state["form"]["smoker"] = smoker
        
        return True
    
    return validate



def show_health_form():
    st.header("Health Information:")
    bp = st.selectbox("Are u high blood?", 
                        options=bool_options,
                        help="Blood pressure readings are consistently at or above 130/80 mm Hg")
    chol = st.selectbox("High cholesterol levels?", 
                    options=bool_options,
                    help="Consistent high cholesterol level of 200 mg/dL")
    stroke = st.selectbox("Have you had a stroke?", 
                    options=bool_options)
    diff_walk = st.selectbox("Any experience difficulty in walking/climbing up stairs?", 
                options=bool_options)
    
    def validate():
        answers = [bp, chol, stroke, diff_walk]
        if any(ans == PLACEHOLDER for ans in answers):
            st.warning("Please select a valid answer for all questions.")
            return False
        st.session_state["form"]["bp"] = bp
        st.session_state["form"]["chol"] = chol
        st.session_state["form"]["stroke"] = stroke
        st.session_state["form"]["diff_walk"] = diff_walk
        return True
        
    return validate
    

def show_summary():
    st.write("Please review your submission before getting your reults.")
    st.subheader("Take note, our model's accuracy has a 87% accuracy; 65% recall score (trained on US 2021 health data)")

    st.write(st.session_state['form'])


    if st.button("Generate Results"):
        progress = st.progress(0)

        st.write("Running random forest... This may take a while...")
        for i in range(100):
            progress.progress(i+1)
            time.sleep(np.random.uniform(0.01,0.5))
        st.success("Completed! Please see your results below:")

        st.session_state['Gen_State'] = 'success'
        if st.session_state.get('Gen_State') == 'success':
            st.image("https://i.makeagif.com/media/2-28-2017/Sc8nWa.gif")