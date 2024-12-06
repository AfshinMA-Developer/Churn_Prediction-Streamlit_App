import os
import joblib
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load models and preprocessor
model_dir = 'models'
data_dir = 'datasets'

preprocessor_path = os.path.join(model_dir, 'churn_preprocessor.joblib')
loaded_preprocessor = joblib.load(preprocessor_path)

model_names = [
    'Ada Boost Classifier',
    'Extra Trees Classifier',
    'Gradient Boosting Classifier',
    'LGBM Classifier', 
    'LogisticRegression',
    'RandomForestClassifier'
    'XGBoost Classifier', 
]
model_paths = {name: os.path.join(model_dir, f"{name.replace(' ', '')}.joblib") for name in model_names}

# Load models safely
models = {}
for name, path in model_paths.items():
    try:
        models[name] = joblib.load(path)
    except Exception as e:
        print(f"Error loading model {name} from {path}: {str(e)}")
        
# Load dataset
data_path = os.path.join(data_dir, 'cleaned_IT_customer_churn.csv')
df = pd.read_csv(data_path)

# Prepare features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Predefined input choices
input_choices = {
    'gender': ['Female', 'Male'],
    'internet_service': ['DSL', 'Fiber optic', 'No'],
    'contract': ['Month-to-month', 'One year', 'Two year'],
    'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

# Pre-computed statistics for default values
stats = df[['tenure', 'MonthlyCharges', 'TotalCharges']].agg(['mean', 'max']).reset_index()
means = stats.loc[0]
maxs = stats.loc[1]

# Metrics calculation function
def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred) * 100,
        'Recall': recall_score(y_true, y_pred) * 100,
        'F1': f1_score(y_true, y_pred) * 100,
        'Precision': precision_score(y_true, y_pred) * 100,
    }

# Prediction and metrics evaluation function
def load_and_predict(sample):
    try:
        sample_trans = loaded_preprocessor.transform(sample)
        X_trans = loaded_preprocessor.transform(X)

        # Using SMOTE to handle class imbalance
        X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_trans, y)

        results = []
        for name, model in models.items():
            churn_pred = model.predict(sample_trans)
            y_resampled_pred = model.predict(X_resampled)
            metrics = calculate_metrics(y_resampled, y_resampled_pred)

            results.append({
                'Model': name,
                'Predicted Churn': 'Yes' if churn_pred[0] == 1 else 'No',
                **metrics,
            })

        return pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

    except Exception as e:
        st.error(f"An error occurred during model loading or prediction: {str(e)}")
        return pd.DataFrame()

# Streamlit UI setup
st.set_page_config(page_title="Churn Prediction App", page_icon="♻️", layout="wide")
st.title("♻️ **Customer Churn Prediction**")
st.subheader("Enter the following information to predict **Churn**.")

# Streamlit form for input
with st.form(key='churn_form'):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("**Gender**", options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
        internet_service = st.selectbox("**Internet Service**", options=input_choices['internet_service'])
        contract = st.selectbox("**Contract**", options=input_choices['contract'])
        payment_method = st.selectbox("**Payment Method**", options=input_choices['payment_method'])
        tenure = st.slider("**Tenure (Months)**", 0, int(maxs['tenure'] * 1.5), int(means['tenure']))
        monthly_charges = st.number_input("**Monthly Charges**", 0.0, float(maxs['MonthlyCharges'] * 1.5), float(means['MonthlyCharges']))
        total_charges = st.number_input("**Total Charges**", 0.0, float(maxs['TotalCharges'] * 1.5), float(means['TotalCharges']))
    with col2:
        st.subheader("Additional Information")

        boolean_inputs = [
            "Senior Citizen", "Partner", "Dependents", "Phone Service",
            "Online Security", "Online Backup", "Device Protection",
            "Multiple Lines", "Tech Support", "Streaming TV", "Streaming Movies", "Paperless Billing"
        ]
        
        # Create radio buttons for additional boolean inputs
        boolean_values = {input_name: st.radio(f"**{input_name}**", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No') for input_name in boolean_inputs}
        

    st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line

    # Handling the predictions when the button is pressed
    if st.form_submit_button(label=':rainbow[Predict Churn]'):
        # Prepare input data for prediction
        input_data = pd.DataFrame([{
            'gender': int(gender),
            'SeniorCitizen': int(boolean_values["Senior Citizen"]),
            'Partner': int(boolean_values["Partner"]),
            'Dependents': int(boolean_values["Dependents"]),
            'tenure': int(tenure),
            'PhoneService': int(boolean_values["Phone Service"]),
            'MultipleLines': int(boolean_values["Multiple Lines"]),
            'InternetService': str(internet_service),
            'OnlineSecurity': int(boolean_values["Online Security"]),
            'OnlineBackup': int(boolean_values["Online Backup"]),
            'DeviceProtection': int(boolean_values["Device Protection"]),
            'TechSupport': int(boolean_values["Tech Support"]),
            'StreamingTV': int(boolean_values["Streaming TV"]),
            'StreamingMovies': int(boolean_values["Streaming Movies"]),
            'Contract': str(contract),
            'PaperlessBilling': int(boolean_values["Paperless Billing"]),
            'PaymentMethod': str(payment_method),
            'MonthlyCharges': float(monthly_charges),
            'TotalCharges': float(total_charges),
        }])

        # Predicting the input data
        results_df = load_and_predict(input_data)
        
        # Displaying results
        if not results_df.empty:
            st.write("### Prediction Results:")
            styled_df = results_df.style.map(lambda x: 'color: green' if x == 'Yes' else 'color: red', subset=['Predicted Churn'])
            st.dataframe(styled_df)

# Disclaimer Section
st.markdown("---")
st.text('''
        >> Customer Churn Prediction App <<
        This Streamlit application predicts customer churn using multiple machine learning models including LGBM, XGBoost, and Gradient Boosting classifiers. 
        Users can input customer information through a user-friendly interface, which includes various fields such as gender, tenure, internet service type, and service usage details.
        
        > Features:
            Input Components: Users can provide data using radio buttons, sliders, and dropdowns to answer questions regarding gender, service preferences, and usage metrics.
            Data Processing: Upon submitting the form, the app processes the input data and transforms it using a pre-trained data preprocessor. 
            It leverages SMOTE to address any class imbalance in the data.
            Prediction: The app runs predictions using the loaded models and calculates performance metrics like accuracy, recall, F1 score, and precision.
            Results Display: The predicted churn status and model performance metrics are displayed in a formatted output for easy interpretation.
        > Usage: Just fill out the information about the customer and click "Predict Churn" to receive insights on whether the customer is likely to churn and how well each model performed.
        > Disclaimer: This application is intended for educational purposes only.
''')