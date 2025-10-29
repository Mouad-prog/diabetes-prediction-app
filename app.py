import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model # CORRECTED: Use top-level keras import
from sklearn.preprocessing import StandardScaler
import os
import joblib # Keep joblib for the scaler
import pickle # Use pickle for base models

# --- Configuration ---
PROJECT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(PROJECT_DIR, 'stacking_nn_meta_learner.h5')
SCALER_PATH = os.path.join(PROJECT_DIR, 'stacking_scaler.pkl')
BASE_MODELS_DIR = os.path.join(PROJECT_DIR, 'base_models')
OPTIMAL_THRESHOLD = 0.260 # From analysis of stacking_nn_threshold_optimization.png and all_models_results.csv

# NOTE: The model uses 15 ENGINEERED features (not the original 21):
# BMI, GenHlth, HighBP, CardioRisk_Score*, Age, Health_Score*, Income, 
# PhysHlth, HighChol, BMI_Category*, DiffWalk, MentHlth, Education, 
# HeartDiseaseorAttack, Age_HighRisk*
# (* = engineered features created from input data)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    """Loads the neural network meta-learner, the scaler, and the base models."""
    try:
        # Load the Stacking Neural Network Meta-Learner (Keras model)
        st.info("Loading neural network meta-learner...")
        meta_learner = load_model(MODEL_PATH)
        st.success("✓ Neural network loaded successfully")
        
        # Load the StandardScaler
        st.info("Loading scaler...")
        try:
            scaler = joblib.load(SCALER_PATH)
            st.success("✓ Scaler loaded successfully")
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            st.warning("Trying to load with pickle instead of joblib...")
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            st.success("✓ Scaler loaded successfully with pickle")
            
        # Load the 4 base models (XGBoost, CatBoost, LightGBM, Logistic Regression)
        base_models = {}
        for name in ['xgb', 'cat', 'lgbm', 'log']:
            st.info(f"Loading {name.upper()} model...")
            model_file = os.path.join(BASE_MODELS_DIR, f'stacking_{name}_base.pkl')
            
            try:
                # Try joblib first
                base_models[name] = joblib.load(model_file)
                st.success(f"✓ {name.upper()} model loaded successfully with joblib")
            except Exception as e1:
                st.warning(f"Joblib failed for {name.upper()}: {e1}")
                try:
                    # Try standard pickle
                    with open(model_file, 'rb') as f:
                        base_models[name] = pickle.load(f)
                    st.success(f"✓ {name.upper()} model loaded successfully with pickle")
                except Exception as e2:
                    st.error(f"Error loading {name.upper()} model: {e2}")
                    raise
                
        return meta_learner, scaler, base_models
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.error(f"DEBUG: Error details: {e}")
        st.error("""
        **Possible solutions:**
        1. Update scikit-learn: `pip install --upgrade scikit-learn`
        2. The model files may need to be regenerated with your current Python environment
        3. Check that all .pkl files are not corrupted
        """)
        st.stop()

meta_learner, scaler, base_models = load_assets()

# --- Feature List (from feature_importance_report.csv) ---
USER_FEATURES = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex',
    'Age', 'Education', 'Income', 'CholCheck'
]

# --- Prediction Function ---
def make_prediction(input_data):
    """
    Prepares the input data with feature engineering, runs base model predictions, 
    and feeds them to the meta-learner.
    """
    
    # Feature Engineering (as done during training)
    # CardioRisk_Score: combination of cardiovascular risk factors
    cardio_risk_score = (
        input_data['HighBP'] + 
        input_data['HighChol'] + 
        input_data['HeartDiseaseorAttack'] + 
        input_data['Stroke']
    )
    
    # Health_Score: combination of general health indicators
    health_score = (
        input_data['GenHlth'] + 
        input_data['PhysHlth'] + 
        input_data['MentHlth']
    )
    
    # BMI_Category: categorize BMI into bins
    bmi = input_data['BMI']
    if bmi < 18.5:
        bmi_category = 0  # Underweight
    elif bmi < 25:
        bmi_category = 1  # Normal
    elif bmi < 30:
        bmi_category = 2  # Overweight
    else:
        bmi_category = 3  # Obese
    
    # Age_HighRisk: binary indicator for high-risk age groups
    # Assuming Age is encoded (e.g., 1-13 scale), high risk if Age >= 9 (roughly 55+)
    age_high_risk = 1 if input_data['Age'] >= 9 else 0
    
    # Create feature vector with engineered features (15 features total)
    # Using proper feature names to avoid warnings
    feature_names = ['BMI', 'GenHlth', 'HighBP', 'CardioRisk_Score', 'Age', 'Health_Score', 
                     'Income', 'PhysHlth', 'HighChol', 'BMI_Category', 'DiffWalk', 
                     'MentHlth', 'Education', 'HeartDiseaseorAttack', 'Age_HighRisk']
    
    feature_values = [
        input_data['BMI'], 
        input_data['GenHlth'], 
        input_data['HighBP'], 
        cardio_risk_score,
        input_data['Age'], 
        health_score,
        input_data['Income'], 
        input_data['PhysHlth'], 
        input_data['HighChol'], 
        bmi_category,
        input_data['DiffWalk'], 
        input_data['MentHlth'], 
        input_data['Education'], 
        input_data['HeartDiseaseorAttack'], 
        age_high_risk
    ]
    
    # Create DataFrame with proper feature names to avoid warnings
    feature_df = pd.DataFrame([feature_values], columns=feature_names)
    
    scaled_features = scaler.transform(feature_df)
    
    base_preds = []
    base_preds.append(base_models['xgb'].predict_proba(scaled_features)[:, 1].reshape(-1, 1))
    base_preds.append(base_models['cat'].predict_proba(scaled_features)[:, 1].reshape(-1, 1))
    base_preds.append(base_models['lgbm'].predict_proba(scaled_features)[:, 1].reshape(-1, 1))
    base_preds.append(base_models['log'].predict_proba(scaled_features)[:, 1].reshape(-1, 1))
    
    level1_input = np.hstack(base_preds)
    
    probability = meta_learner.predict(level1_input)[0][0]
    
    prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
    
    return prediction, probability

# --- Streamlit App Layout ---
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("🩺 Diabetes Prediction App")
st.markdown("A machine learning application to predict the risk of diabetes based on health indicators, using a **Stacking Ensemble Model** with an optimized threshold.")

st.sidebar.header("User Input Features")

input_data = {}

st.sidebar.subheader("Health Conditions (0=No, 1=Yes)")
input_data['HighBP'] = st.sidebar.selectbox('High Blood Pressure', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['HighChol'] = st.sidebar.selectbox('High Cholesterol', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['Smoker'] = st.sidebar.selectbox('Smoked at least 100 cigarettes in life', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['Stroke'] = st.sidebar.selectbox('Had a Stroke', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['HeartDiseaseorAttack'] = st.sidebar.selectbox('Coronary Heart Disease or Myocardial Infarction', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['DiffWalk'] = st.sidebar.selectbox('Difficulty Walking or Climbing Stairs', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['AnyHealthcare'] = st.sidebar.selectbox('Any Healthcare Coverage', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['NoDocbcCost'] = st.sidebar.selectbox('Could not see a doctor due to cost', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['CholCheck'] = st.sidebar.selectbox('Cholesterol Check in past 5 years', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')

st.sidebar.subheader("Lifestyle")
input_data['PhysActivity'] = st.sidebar.selectbox('Physical Activity in past 30 days', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['Fruits'] = st.sidebar.selectbox('Consume Fruit 1 or more times per day', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['Veggies'] = st.sidebar.selectbox('Consume Vegetables 1 or more times per day', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
input_data['HvyAlcoholConsump'] = st.sidebar.selectbox('Heavy Alcohol Consumption (men >14, women >7 drinks/week)', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')

st.sidebar.subheader("Metrics & Demographics")
input_data['BMI'] = st.sidebar.slider('BMI', 12.0, 50.0, 25.0)
input_data['GenHlth'] = st.sidebar.selectbox('General Health (1=Excellent to 5=Poor)', range(1, 6))
input_data['MentHlth'] = st.sidebar.slider('Days of poor mental health in past 30 days', 0, 30, 0)
input_data['PhysHlth'] = st.sidebar.slider('Days of poor physical health in past 30 days', 0, 30, 0)
input_data['Sex'] = st.sidebar.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x==1 else 'Female')

age_map = {i: f'{18 + (i-1)*5}-{24 + (i-1)*5}' for i in range(1, 13)}
age_map[13] = '80+'
input_data['Age'] = st.sidebar.selectbox('Age Category', list(age_map.keys()), format_func=lambda x: age_map[x])

education_map = {1: 'Never attended', 2: 'Elementary', 3: 'Some High School', 4: 'High School Grad', 5: 'Some College', 6: 'College Grad'}
input_data['Education'] = st.sidebar.selectbox('Education Level', list(education_map.keys()), format_func=lambda x: education_map[x])

income_map = {1: '<$10k', 2: '$10k-$15k', 3: '$15k-$20k', 4: '$20k-$25k', 5: '$25k-$35k', 6: '$35k-$50k', 7: '$50k-$75k', 8: '>$75k'}
input_data['Income'] = st.sidebar.selectbox('Income Category', list(income_map.keys()), format_func=lambda x: income_map[x])

if st.sidebar.button('Predict Diabetes Risk'):
    with st.spinner('Calculating prediction...'):
        prediction, probability = make_prediction(input_data)
    
    st.header("Prediction Result")
    
    if prediction == 1:
        st.error(f"High Risk of Diabetes (Prediction: **Positive**)")
    else:
        st.success(f"Low Risk of Diabetes (Prediction: **Negative**)")
        
    st.subheader("Details")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Predicted Probability", f"{probability:.4f}")
    col2.metric("Optimal Threshold", f"{OPTIMAL_THRESHOLD:.4f}")
    
    if probability >= OPTIMAL_THRESHOLD:
        st.warning(f"The predicted probability ({probability:.4f}) is greater than or equal to the optimal threshold ({OPTIMAL_THRESHOLD:.4f}), resulting in a **Positive** prediction.")
    else:
        st.info(f"The predicted probability ({probability:.4f}) is less than the optimal threshold ({OPTIMAL_THRESHOLD:.4f}), resulting in a **Negative** prediction.")
        
    st.markdown("--")
    st.subheader("Model Information")
    st.markdown(f"The prediction was made using a **Stacking Ensemble Model** (XGBoost, CatBoost, LightGBM, Logistic Regression) with a **Neural Network** as the meta-learner.")
    st.markdown(f"The model uses an **optimized decision threshold of {OPTIMAL_THRESHOLD}** to maximize the F1-Score, which is crucial for balancing Precision and Recall in this imbalanced classification problem.")
else:
    st.info("Adjust the features in the sidebar and click 'Predict Diabetes Risk' to get a prediction.")
st.sidebar.markdown("--")
st.sidebar.caption("Model Feature Importance (Top 5)")
st.sidebar.image(os.path.join(PROJECT_DIR, "assets/feature_importance_top5.png"), width="stretch")
