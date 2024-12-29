from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the scaler and ensemble model
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
model_path = os.path.join(os.getcwd(), 'ensemble_model.pkl')

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(model_path, 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

# Mapping for categorical variables
payment_min_map = {'No': 0, 'Yes': 1}
credit_mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data for the selected features only
        num_bank_accounts = float(request.form['num_bank_accounts'])
        num_credit_card = float(request.form['num_credit_card'])
        interest_rate = float(request.form['interest_rate'])
        num_of_loan = float(request.form['num_of_loan'])
        delay_from_due_date = float(request.form['delay_from_due_date'])
        num_of_delayed_payment = float(request.form['num_of_delayed_payment'])
        num_credit_inquiries = float(request.form['num_credit_inquiries'])
        outstanding_debt = float(request.form['outstanding_debt'])
        credit_history_age = float(request.form['credit_history_age'])
        payment_of_min_amount = payment_min_map[request.form['payment_of_min_amount']]

        # Create the feature array for the selected features
        feature_array = np.array([[num_bank_accounts, num_credit_card, interest_rate, num_of_loan,
                                   delay_from_due_date, num_of_delayed_payment, num_credit_inquiries,
                                   outstanding_debt, credit_history_age, payment_of_min_amount]])

        # Scale the input data
        feature_array_scaled = scaler.transform(feature_array)

        # Make prediction
        prediction = ensemble_model.predict(feature_array_scaled)

        # Map prediction back to Credit Score category
        credit_score_map = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        predicted_score = credit_score_map[prediction[0]]

        return render_template('result.html', prediction=predicted_score)

if __name__ == '__main__':
    app.run(debug=True)
