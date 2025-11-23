import streamlit as st
import time
import numpy as np
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb 

gender_mapping = {"Male":1,
                  "Female":0}
bool_mapping = {"Yes":1,
                "No":0}
education_mapping = {
    1: "Never attended school / Kindergarten only",
    2: "Grades 1–8 (Elementary)",
    3: "Grades 9–11 (Some high school)",
    4: "High school graduate",
    5: "Some college or technical school (No degree)",
    6: "College graduate (4 years or more)"
}



age_group_mapping = {
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+"
}

income_mapping = {
    1: "Less than $10,000",
    2: "$10,000 - $15,000",
    3: "$15,000 - $20,000",
    4: "$20,000 - $25,000",
    5: "Less than $35,000",
    6: "Less than $50,000",
    7: "Less than $75,000",
    8: "$75,000 or more"
}


["-- Select Gender --"]
bool_list = ['bp','chol','chol_check','smoker','stroke','heart_disease']
gender_options =  list(gender_mapping.keys())
PLACEHOLDER = "-- Select Answer --"
bool_options = list(bool_mapping.keys())

def preprocess_input(form_data):
    processed = {}

    processed["height"] = form_data.get("height")
    processed["weight"] = form_data.get("weight")
    processed["bmi"] = form_data.get("bmi")
    processed["menthlth"] = form_data.get("menthlth")
    processed["physHlth"] = form_data.get("physHlth")
    processed["general_health"] = form_data.get("general_health")
    processed["education_code"] = form_data.get("educ")
    
    label = form_data.get("income")

    # reverse mapping: label -> code
    reverse_income = {v: k for k, v in income_mapping.items()}

    processed["income"] = reverse_income.get(label, None)

    gender_value = form_data.get("gender")
    if isinstance(gender_value, str):
        processed["gender"] = gender_mapping.get(gender_value, None)
    else:
        processed["gender"] = gender_value

    bool_fields = [
        "fruit", "vegetable", "phys", "alcohol", "smoker",
        "bp", "chol_check", "chol", "stroke", "diff_walk", 
        "heart_disease","healthcare","nodoc_cost"
    ]

    for field in bool_fields:
        value = form_data.get(field)
        processed[field] = bool_mapping.get(value, value)

    age = form_data.get("age")

    if age:
        if 25 <= age <= 29:
            processed["age_group"] = 2
        elif 30 <= age <= 34:
            processed["age_group"] = 3
        elif 35 <= age <= 39:
            processed["age_group"] = 4
        elif 40 <= age <= 44:
            processed["age_group"] = 5
        elif 45 <= age <= 49:
            processed["age_group"] = 6
        elif 50 <= age <= 54:
            processed["age_group"] = 7
        elif 55 <= age <= 59:
            processed["age_group"] = 8
        elif 60 <= age <= 64:
            processed["age_group"] = 9
        elif 65 <= age <= 69:
            processed["age_group"] = 10
        elif 70 <= age <= 74:
            processed["age_group"] = 11
        elif 75 <= age <= 79:
            processed["age_group"] = 12
        elif age >= 80:
            processed["age_group"] = 13
        else:
            processed["age_group"] = None

    return processed


def map_to_model_features(data):
    """
    Convert processed user input into the exact feature vector required by the model.
    """

    feature_dict = {
        "HighBP": data.get("bp"),
        "HighChol": data.get("chol"),
        "CholCheck": data.get("chol_check"),
        "BMI": data.get("bmi"),
        "Smoker": data.get("smoker"),
        "Stroke": data.get("stroke"),
        "HeartDiseaseorAttack": data.get("heart_disease"),  
        "PhysActivity": data.get("phys"),
        "Fruits": data.get("fruit"),
        "Veggies": data.get("vegetable"),
        "HvyAlcoholConsump": data.get("alcohol"),
        "GenHlth": data.get("general_health"),
        "MentHlth": data.get("menthlth"),
        "PhysHlth": data.get("physHlth"),
        "DiffWalk": data.get("diff_walk"),
        "Sex": data.get("gender"),
        "Age": data.get("age_group"),
        "Education": data.get("education_code", None),
        "Income": data.get("income", None),
        "AnyHealthcare": data.get("healthcare"),
        "NoDocbcCost": data.get("nodoc_cost")
    }

    # Convert to DataFrame in correct order
    feature_order = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']

    return pd.DataFrame([feature_dict])[feature_order]


def load_saved_value(field_name, options):
    """Return index of saved value if available, else None."""
    saved_value = st.session_state.get("form", {}).get(field_name, None)
    if saved_value in options:
        return options.index(saved_value)
    return None



def next_form(session_key):
    keys = ['name_info','bio_info','lifestyle','health_info','wellbeing_info','summary']

    idx = keys.index(session_key)

    if idx == len(keys) - 1:  # last page
        st.session_state["current_form"] = "summary"
    else:
        st.session_state["current_form"] = keys[idx + 1]

def previous_form(session_key):
    keys = ['name_info','bio_info','lifestyle','health_info','wellbeing_info','summary']

    idx = keys.index(session_key)

    if idx == 0:  # already first page
        st.session_state["current_form"] = keys[0]
    else:
        st.session_state["current_form"] = keys[idx - 1]


def show_name_form():
    saved_value = st.session_state.get("form", {}).get("name", None)
    name = st.text_input("Enter your name",value=saved_value)

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
    
    saved_value = st.session_state.get("form", {}).get("age", 25)
    saved_value = int(saved_value) 
    age = st.slider("Select your age", min_value=25, max_value=100, value=saved_value, step=1)

    st.write(f"What is your assigned sex at birth?")

    options = list(gender_mapping.keys())   # ["Male", "Female"]

    saved_code = st.session_state.get("form", {}).get("gender", None)

    # Convert saved code → label
    if saved_code in gender_mapping.values():
        # find the key (label) whose value matches saved_code
        saved_label = [k for k,v in gender_mapping.items() if v == saved_code][0]
        default_index = options.index(saved_label)
    else:
        default_index = None
    gender = st.selectbox("Gender:", options=gender_options,
                          index=default_index)
    
    st.write('Please input your height and weight:')

    saved_value = st.session_state.get("form", {}).get("height", 1.0)
    saved_value = float(saved_value)
    height = st.number_input(
        "Height (cm)",
        step=1.0,
        value=saved_value,
        min_value=1.0
    )

    saved_value = st.session_state.get("form", {}).get("weight", 1.0)
    saved_value = float(saved_value)
    weight = st.number_input(
        "Weight (kg)",
        format="%0.1f",
        value=saved_value,
        min_value=1.0
    )
    bmi = weight / ((height / 100) ** 2)

    saved_code = st.session_state.get("form", {}).get("educ", None)
    if saved_code in education_mapping:
        saved_label = education_mapping[saved_code]
        default_index = list(education_mapping.values()).index(saved_label)
    else:
        default_index = None
    educ = st.selectbox(
        "Educational Level",
        list(education_mapping.values()),
        index=default_index)
    
    def validate():
        if height <= 0 or weight <= 0 or gender is None or educ is None:
            st.warning("Please fill in all fields.")
            return False
        
        if gender == "-- Select Gender --" or gender is None:
            st.warning("Please choose a gender")
            return False
        st.session_state["form"]["height"] = height
        st.session_state["form"]["weight"] = weight
        st.session_state["form"]["bmi"] = bmi
        st.session_state["form"]["age"] = age
        st.session_state["form"]["gender"] = gender_mapping[gender]
        st.session_state["form"]["educ"] = [key for key, val in education_mapping.items() if val == educ][0]
        return True
    
    return validate

def show_lifestyle_form():
    st.header("Lifestyle Information: ")
    fruit = st.selectbox("Do you eat any fruit/s at least once (1) per day?",                          
                         options=bool_options,
                         index=load_saved_value("fruit", bool_options))
    
    vegetable = st.selectbox("Do you eat any vegetable/s at least once (1) per day?",
                             options=bool_options,
                             index=load_saved_value("vegetable", bool_options))
    
    phys = st.selectbox("Have you done any physical activity in the past 30 days (not including job)?",
                       options=bool_options,
                       index=load_saved_value("phys", bool_options))
    
    gender_value = st.session_state.form.get("gender", None)
    if gender_value == 1:
        num_glasses = 14
    else:
        num_glasses = 7  
    alcohol = st.selectbox(f"Do you drink more than (>=) {num_glasses} glasses of alcohol per week?",
                           options=bool_options,
                           index=load_saved_value("alcohol", bool_options))

    smoker = st.selectbox(f"Have you smoked at least 100 cigarettes in your entire life?",
                           options=bool_options,
                           help="Note: 5 packs = 100 cigarettes",
                           index=load_saved_value("smoker", bool_options))
    def validate():
        answers = [fruit, vegetable, phys, alcohol, smoker]
        if any(ans == PLACEHOLDER or ans is None for ans in answers):
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
    bp = st.selectbox("Do you have high blood pressure?", 
                        options=bool_options,
                        help="Blood pressure readings are consistently at or above 130/80 mm Hg",
                        index=load_saved_value("bp", bool_options))
    chol_check = st.selectbox("Have you had your cholesterol levels checked in the last 5 years?", 
                    options=bool_options,
                    index=load_saved_value("chol_check", bool_options)) 
    chol = st.selectbox("Do you have high cholesterol levels?", 
                    options=bool_options,
                    help="Consistent high cholesterol level of 200 mg/dL",
                    index=load_saved_value("chol", bool_options))
    stroke = st.selectbox("Have you had a stroke?", 
                    options=bool_options,
                    index=load_saved_value("stroke", bool_options))
    heart_disease = st.selectbox("Do you have any heart conditions or experienced heart attack?", 
                    options=bool_options, 
                    help="Disease/Condition may be Coronary Heart disease or Myocardial infarction",
                    index=load_saved_value("heart_disease", bool_options))
    diff_walk = st.selectbox("Any experience difficulty in walking/climbing up stairs?", 
                options=bool_options,
                index=load_saved_value("diff_walk", bool_options))
    
    def validate():
        answers = [bp, chol, stroke, diff_walk, heart_disease]
        if any(ans == PLACEHOLDER or ans is None for ans in answers):
            st.warning("Please select a valid answer for all questions.")
            return False
        st.session_state["form"]["bp"] = bp
        st.session_state["form"]["chol_check"] = chol_check
        st.session_state["form"]["chol"] = chol
        st.session_state["form"]["stroke"] = stroke
        st.session_state["form"]["heart_disease"] = heart_disease
        st.session_state["form"]["diff_walk"] = diff_walk
        return True
        
    return validate


def show_wellbeing_form():
    st.header("General Well-being and Other Information:")

    saved_value = st.session_state.get("form", {}).get("general_health", None)
    general_health = st.radio(
        "How would you rate your general health overall?",
        options=[1, 2, 3, 4, 5],
        index=None if saved_value is None else [1, 2, 3, 4, 5].index(saved_value),
        format_func=lambda x: {
            1: "1 - Excellent",
            2: "2 - Very good",
            3: "3 - Fair",
            4: "4 - Poor",
            5: "5 - Very Poor"
        }[x],
    )

    saved_value = st.session_state.get("form", {}).get("menthlth", 0)
    menthlth = st.slider(
        "How many days during the past 30 days was your **mental health not good**?",
        min_value=0,
        max_value=30,
        value=saved_value,
        help="Includes stress, depression, emotional problems."
    )

    saved_value = st.session_state.get("form", {}).get("physHlth", 0)
    physHlth = st.slider(
        "How many days during the past 30 days was your **physical health not good**?",
        min_value=0,
        max_value=30,
        value=saved_value,
        help="Includes illness and injury."
    )

    healthcare = st.selectbox("Do you have any kind of healthcare coverage?", 
            options=bool_options,
            help="Includes: Health Insurance, Prepaid plans, HMO, etc.",
            index=load_saved_value("healthcare", bool_options))

    nodoc_cost = st.selectbox("Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?", 
            options=bool_options,
            index=load_saved_value("nodoc_cost", bool_options))
    

    saved_income_code = st.session_state.get("form", {}).get("income", None)

    if saved_income_code in income_mapping:
        saved_label = income_mapping[saved_income_code]
        income_index = list(income_mapping.values()).index(saved_label)
    else:
        income_index = 0

    income = st.selectbox(
        "Annual Household Income (USD)",
        options=list(income_mapping.values()),
        index=income_index,
        help="This information helps estimate socioeconomic risk factors for diabetes."
    )

    
    def validate():
        answers = [general_health, menthlth, physHlth, healthcare, nodoc_cost, income]
        if any(ans == '' or ans is None for ans in answers):
            st.warning("Please select a valid answer for all questions.")
            return False
        st.session_state["form"]["menthlth"] = menthlth
        st.session_state["form"]["general_health"] = general_health
        st.session_state["form"]["physHlth"] = physHlth
        st.session_state["form"]["healthcare"] = healthcare
        st.session_state["form"]["nodoc_cost"] = nodoc_cost
        st.session_state["form"]["income"] = income
        return True
        
    return validate
    

def show_summary():
    st.write("Please review your submission before getting your results.")
    st.subheader("Note: Model Performance and classification")
    st.write("The model performs well in distinguishing between people with diabetes and without (0.8293 AUC ROC). Based on testing, it does well in catching people who actually have diabetes (83% Recall), but it sometimes might flag people as diabetic when they are not. The model is designed to minimize missed cases of diabetes. It is still best to consult your physician for symptoms or tests.")

    st.write(st.session_state['form'])

    if st.button("Generate Results"):
        st.session_state["current_form"] = "results"
        st.rerun()
        
        


# def show_results():

#     # Preprocess
#     processed_data = preprocess_input(st.session_state['form'])
#     model_input = xgb.DMatrix(map_to_model_features(processed_data))
    
#     # Load model
#     model = joblib.load("DiabPred_XGB_SMOTE.pkl")

#     # Predict
#     prediction = model.predict(model_input)[0]
#     probability = model.predict(model_input)[0]

#     prediction = int(probability >= 0.5)

#     # Map prediction to a friendly message
#     if prediction == 1:
#         st.success(f"Based on the model, you are **predicted to have diabetes**.")
#         st.write(f"Predicted probability: **{probability:.2f}**")
#         st.info("It’s important to consult a healthcare professional as soon as possible. Please visit your nearest doctor for a proper check-up.")
#     else:
#         st.success(f"Based on the model, you are **predicted to not have diabetes**.")
#         st.write(f"Predicted probability of diabetes: **{probability:.2f}**")
#         st.info("Even if the result looks good, if you have symptoms or concerns, it’s always best to check with your physician for confirmation.")


#     # SHAP EXPLAINER (new API)
#     explainer = shap.TreeExplainer(model)
#     shap_values_obj = explainer(model_input)

#     raw = shap_values_obj.values

#     # Convert to numpy
#     raw = np.array(raw)

#     # Case 1: (1, n_features)
#     if raw.ndim == 2:
#         shap_vals = raw[0]

#     # Case 2: (1, n_features, 1)
#     elif raw.ndim == 3 and raw.shape[-1] == 1:
#         shap_vals = raw[0, :, 0]

#     # Case 3: (n_outputs, n_features)
#     elif raw.ndim == 2 and raw.shape[0] == 2:
#         shap_vals = raw[1]   # class 1

#     # Case 4: (1, n_features, 2)
#     elif raw.ndim == 3 and raw.shape[-1] == 2:
#         shap_vals = raw[0, :, 1]  # class 1

#     # Fallback: flatten
#     else:
#         shap_vals = raw.reshape(-1)

    
#     df_input = model_input

#     # Build SHAP summary table
#     shap_df = pd.DataFrame({
#         "feature": df_input.columns,
#         "shap_value": shap_vals,
#         "abs_shap": np.abs(shap_vals)
#     }).sort_values("abs_shap", ascending=False)


#     # SHAP Bar Plot (Streamlit-safe)
#     st.write("### SHAP Bar Plot (Feature Importance)")
#     fig, ax = plt.subplots()
#     shap.summary_plot(
#         shap_vals.reshape(1, -1),
#         df_input,
#         plot_type="bar",
#         show=False
#     )
#     st.pyplot(fig)

#     # Optional simple interpretation
#     top_feature = shap_df.iloc[0]
#     tendency = "increases" if top_feature["shap_value"] > 0 else "decreases"

#     st.info(
#         f"**Most influential feature:** `{top_feature['feature']}`\n\n"
#         f"It **{tendency}** the probability of diabetes by "
#         f"**{abs(top_feature['shap_value']):.3f}** SHAP units."
#     )






def show_results():
    st.subheader("Note: Model Performance and classification")
    st.write("The model performs well in distinguishing between people with diabetes and without (0.8293 AUC ROC). Based on testing, it does well in catching people who actually have diabetes (83% Recall), but it sometimes might flag people as diabetic when they are not. The model is designed to minimize missed cases of diabetes. It is still best to consult your physician for symptoms or tests.")


    processed_data = preprocess_input(st.session_state['form'])
    df_input = map_to_model_features(processed_data)  # Keep as DataFrame for SHAP/feature names
    model_input = xgb.DMatrix(df_input)  # for raw Booster prediction


    model = joblib.load("DiabPred_XGB_SMOTE.pkl")


    # Raw Booster returns probabilities if trained with 'binary:logistic'
    probability = model.predict(model_input)[0]  # probability of class 1
    prediction = int(probability >= 0.5)


    if prediction == 1:
        st.success("Based on the model, you are **predicted to have diabetes**.")
        st.write(f"Predicted probability: **{probability:.2f}**")
        st.info("It’s important to consult a healthcare professional as soon as possible. "
                "Please visit your nearest doctor for a proper check-up.")
    else:
        st.success("Based on the model, you are **predicted to not have diabetes**.")
        st.write(f"Predicted probability of diabetes: **{probability:.2f}**")
        st.info("Even if the result looks good, if you have symptoms or concerns, "
                "it’s always best to check with your physician for confirmation.")


    explainer = shap.TreeExplainer(model)
    shap_values_obj = explainer(model_input)

    # Get SHAP values for class 1
    shap_vals_raw = np.array(shap_values_obj.values)

    # Handle various SHAP array shapes
    if shap_vals_raw.ndim == 2:  # (1, n_features)
        shap_vals = shap_vals_raw[0]
    elif shap_vals_raw.ndim == 3 and shap_vals_raw.shape[-1] == 1:
        shap_vals = shap_vals_raw[0, :, 0]
    elif shap_vals_raw.ndim == 3 and shap_vals_raw.shape[-1] == 2:
        shap_vals = shap_vals_raw[0, :, 1]  # class 1
    else:
        shap_vals = shap_vals_raw.reshape(-1)

    # Build SHAP summary DataFrame
    shap_df = pd.DataFrame({
        "feature": df_input.columns,
        "shap_value": shap_vals,
        "abs_shap": np.abs(shap_vals)
    }).sort_values("abs_shap", ascending=False)


    st.write("### Feature Importance (SHAP)")
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_input.columns)*0.3)))
    shap.summary_plot(
        shap_vals.reshape(1, -1),
        df_input,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)


    top_feature = shap_df.iloc[0]
    tendency = "increases" if top_feature["shap_value"] > 0 else "decreases"

    st.info(
        f"**Most influential feature:** `{top_feature['feature']}`\n\n"
        f"It **{tendency}** the probability of diabetes by "
        f"**{abs(top_feature['shap_value']):.3f} SHAP units**.\n\n"
        "The SHAP value shows how much this feature contributed to the prediction. "
        "Positive SHAP values push the prediction towards diabetes, negative values push it away."
    )

    # Optionally show top 5 features
    st.write("#### Top 5 features contributing to prediction")
    st.table(shap_df.head(5)[["feature", "shap_value"]])