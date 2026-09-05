from pathlib import Path
import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️")
st.title("Tourism Package Purchase Predictor")
st.caption("Decision support for prioritizing customers; a prediction is not a guarantee.")
bundle = joblib.load(Path(__file__).with_name("tourism_model.joblib"))
model, threshold, features = bundle["model"], bundle["threshold"], bundle["features"]
defaults = {"Age":35,"TypeofContact":"Self Enquiry","CityTier":1,"DurationOfPitch":15,
"Occupation":"Salaried","Gender":"Female","NumberOfPersonVisiting":2,"NumberOfFollowups":3,
"ProductPitched":"Basic","PreferredPropertyStar":3,"MaritalStatus":"Single","NumberOfTrips":2,
"Passport":1,"PitchSatisfactionScore":3,"OwnCar":0,"NumberOfChildrenVisiting":1,
"Designation":"Executive","MonthlyIncome":25000}
with st.form("customer"):
    values = {}
    categorical = {"TypeofContact":["Self Enquiry","Company Invited"],
      "Occupation":["Salaried","Free Lancer","Small Business","Large Business"],
      "Gender":["Female","Male"],"ProductPitched":["Basic","Standard","Deluxe","Super Deluxe","King"],
      "MaritalStatus":["Single","Married","Divorced"],
      "Designation":["Executive","Manager","Senior Manager","AVP","VP"]}
    for name in features:
        if name in categorical: values[name] = st.selectbox(name, categorical[name])
        elif name in ["Passport","OwnCar"]: values[name] = st.selectbox(name,[0,1],index=int(defaults[name]))
        else: values[name] = st.number_input(name, value=float(defaults[name]))
    submitted = st.form_submit_button("Predict")
if submitted:
    probability = float(model.predict_proba(pd.DataFrame([values],columns=features))[:,1][0])
    st.metric("Estimated purchase probability", f"{probability:.1%}")
    st.success("Prioritize for contact") if probability >= threshold else st.info("Lower priority")
    st.caption(f"Decision threshold: {threshold:.2f}")
