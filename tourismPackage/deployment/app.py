import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="kesavak/tourism_package_model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("🧳 Tourism Package Purchase Predictor")
st.write("""
This application predicts whether a customer will purchase a **tourism package** 
based on demographics, travel preferences, and pitch details.
Enter customer information below to get purchase prediction.
""")

# Tourism-specific input widgets
st.header("Customer Demographics")
age = st.slider("Age", 15, 70, 35)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Big Business", "Executive"])
monthly_income = st.number_input("Monthly Income", 10000, 1000000, 50000, step=5000)

st.header("Travel Preferences")
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 5, 0)
number_of_trips = st.number_input("Past Number of Trips", 0, 20, 2)
passport = st.selectbox("Has Passport", ["No", "Yes"])
own_car = st.selectbox("Owns Car", ["No", "Yes"])

st.header("Pitch Details")
typeof_contact = st.selectbox("Type of Contact", ["Cold Outreach", "Self Enquiry"])
duration_of_pitch = st.slider("Duration of Pitch (days)", 1, 30, 7)
number_of_followups = st.slider("Number of Follow-ups", 0, 15, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Premium", "Super Premium"])
preferred_property_star = st.slider("Preferred Hotel Star Rating", 1, 5, 3)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
designation = st.selectbox("Customer Designation", ["Manager", "Senior Manager", "AVP", "VP", "Director", "SVP"])

# Assemble input into DataFrame (must match EXACT training feature order)
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Predict button
if st.button("🔮 Predict Package Purchase", type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("🎉 **Customer WILL purchase tourism package**")
    else:
        st.warning("❌ **Customer will NOT purchase tourism package**")
    
    st.info(f"Purchase Probability: **{probability:.1%}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", "Will Buy" if prediction == 1 else "Will Not Buy")
    with col2:
        st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
