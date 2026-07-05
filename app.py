from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and feature columns
model = joblib.load('model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Collect form values -- matches the new leak-free feature set
        input_data = {
            'CODE_GENDER': int(request.form['CODE_GENDER']),
            'FLAG_OWN_CAR': int(request.form['FLAG_OWN_CAR']),
            'FLAG_OWN_REALTY': int(request.form['FLAG_OWN_REALTY']),
            'AMT_INCOME_TOTAL': float(request.form['AMT_INCOME_TOTAL']),
            'NAME_INCOME_TYPE': int(request.form['NAME_INCOME_TYPE']),
            'NAME_EDUCATION_TYPE': int(request.form['NAME_EDUCATION_TYPE']),
            'NAME_FAMILY_STATUS': int(request.form['NAME_FAMILY_STATUS']),
            'NAME_HOUSING_TYPE': int(request.form['NAME_HOUSING_TYPE']),
            'FLAG_MOBIL': int(request.form['FLAG_MOBIL']),
            'FLAG_WORK_PHONE': int(request.form['FLAG_WORK_PHONE']),
            'FLAG_PHONE': int(request.form['FLAG_PHONE']),
            'FLAG_EMAIL': int(request.form['FLAG_EMAIL']),
            'AGE': int(request.form['AGE']),
            'YEARS_EMPLOYED': int(request.form['YEARS_EMPLOYED']),
            'FAMILY_MEMBERS': float(request.form['FAMILY_MEMBERS']),
            'CHILDREN': float(request.form['CHILDREN']),
            'TOTAL_MONTHS': int(request.form['TOTAL_MONTHS']),
            'WORST_MONTHS_BALANCE': int(request.form['WORST_MONTHS_BALANCE']),
        }

        # Align with training feature columns (order matters)
        input_df = [input_data.get(col, 0) for col in feature_columns]
        input_array = np.array(input_df).reshape(1, -1)

        # Predict
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0]  # [P(class0), P(class1)]

        if prediction == 1:
            result_text = "Credit Card Approved"
            result_class = "approved"
            confidence = round(proba[1] * 100)
        else:
            result_text = "Credit Card Rejected"
            result_class = "rejected"
            confidence = round(proba[0] * 100)

        return render_template('result.html',
                               prediction=result_text,
                               result_class=result_class,
                               confidence=confidence)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)