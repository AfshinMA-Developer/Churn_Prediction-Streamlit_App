import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import gradio as gr

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
    'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    'others' : ['No', 'Yes']
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
        'F1 Score': f1_score(y_true, y_pred) * 100,
        'Precision': precision_score(y_true, y_pred) * 100,
    }

# Prediction and metrics evaluation function
def load_and_predict(
        gender, internet_service, contract, payment_method, tenure, monthly_charges, total_charges, 
        senior_citizen, partner, dependents, phone_service, multiple_lines, online_security, online_backup, 
        device_protection, tech_support, streaming_tv, streaming_movies, paperless_billing):
    
    # Ensure inputs are not None
    try:
        sample = {
            'gender': int(gender == 'Male'),
            'SeniorCitizen': int(senior_citizen == 'Yes'),
            'Partner': int(partner == 'Yes'),
            'Dependents': int(dependents == 'Yes'),
            'tenure': int(tenure),
            'PhoneService': int(phone_service == 'Yes'),
            'MultipleLines': int(multiple_lines == 'Yes'),
            'InternetService': str(internet_service),
            'OnlineSecurity': int(online_security == 'Yes'),
            'OnlineBackup': int(online_backup == 'Yes'),
            'DeviceProtection': int(device_protection == 'Yes'),
            'TechSupport': int(tech_support == 'Yes'),
            'StreamingTV': int(streaming_tv == 'Yes'),
            'StreamingMovies': int(streaming_movies == 'Yes'),
            'Contract': str(contract),
            'PaperlessBilling': int(paperless_billing == 'Yes'),
            'PaymentMethod': str(payment_method),
            'MonthlyCharges': float(monthly_charges),
            'TotalCharges': float(total_charges)
        }
        
        sample_df = pd.DataFrame([sample])
        sample_trans = loaded_preprocessor.transform(sample_df)
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

        return pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    except Exception as e:
        return f"An error occurred during model loading or prediction: {str(e)}"
    
# Gradio Interface setup
input_components = [
    gr.Radio(label="Gender", choices=input_choices['gender'], value=input_choices['gender'][0]),
    gr.Dropdown(label="Internet Service", choices=input_choices['internet_service'], value=input_choices['internet_service'][0]),
    gr.Dropdown(label="Contract", choices=input_choices['contract'], value=input_choices['contract'][0]),
    gr.Dropdown(label="Payment Method", choices=input_choices['payment_method'], value=input_choices['payment_method'][0]),
    gr.Slider(label="Tenure (Months)", minimum=0, maximum=int(maxs['tenure'] * 1.5), value=int(means['tenure'])),
    gr.Number(label="Monthly Charges", minimum=0.0, maximum=float(maxs['MonthlyCharges'] * 1.5), value=float(means['MonthlyCharges'])),
    gr.Number(label="Total Charges", minimum=0.0, maximum=float(maxs['TotalCharges'] * 1.5), value=float(means['TotalCharges'])),
    gr.Radio(label="Senior Citizen", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Partner", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Dependents", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Phone Service", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Multiple Lines", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Online Security", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Online Backup", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Device Protection", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Tech Support", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Streaming TV", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Streaming Movies", choices=input_choices['others'], value=input_choices['others'][0]),
    gr.Radio(label="Paperless Billing", choices=input_choices['others'], value=input_choices['others'][0]),
]

output_component = gr.DataFrame()

# Launching the Gradio Interface
gr.Interface(
    fn=load_and_predict, 
    inputs=input_components,
    outputs=output_component,
    title="♻️ Customer Churn Prediction",
    description="Enter the following information to predict customer churn.",
    flagging_mode="never"  # Replacing allow_flagging with flagging_mode
).launch()