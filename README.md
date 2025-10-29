# 🩺 Diabetes Prediction App

This is a Streamlit application designed to predict the risk of diabetes based on user-provided health and lifestyle indicators.

## 🧠 Model Used

The core of this application is a sophisticated **Stacking Ensemble Model**:

- **Base Learners:** XGBoost, CatBoost, LightGBM, and Logistic Regression.
- **Meta-Learner:** A small Neural Network that takes the predictions of the base learners as input to make the final decision.

This model was selected for its superior performance, particularly its high F1-Score and ROC-AUC, achieved through an **optimized decision threshold of 0.260**.

## 📁 Project Structure

```
diabetes-prediction-app/
├── app.py                     # Streamlit main script
├── stacking_nn_meta_learner.h5  # The Neural Network meta-learner
├── stacking_scaler.pkl          # The StandardScaler object
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── base_models/               # Directory for the base learners
│   ├── stacking_xgb_base.pkl
│   ├── stacking_cat_base.pkl
│   ├── stacking_lgbm_base.pkl
│   └── stacking_log_base.pkl
└── assets/                    # Optional: Images
    └── feature_importance_top5.png
```
