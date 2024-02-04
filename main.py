from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)


dummy_dataset = np.array([
    [25, 0, 120, 70, 0],
    [35, 1, 130, 75, 0],
    [45, 0, 140, 80, 0],
    [55, 1, 160, 110, 1],  
    [20, 0, 140, 60, 1],   
    [40, 1, 155, 105, 1],  
])

X = dummy_dataset[:, :-1]
y = dummy_dataset[:, -1]

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form.get('age'))
        gender = int(request.form.get('gender'))
        blood_pressure = int(request.form.get('blood_pressure'))
        heart_rate = int(request.form.get('heart_rate'))
    except ValueError:
        return jsonify({'error': 'Invalid input. Please provide valid integer values for age, gender, blood pressure, and heart rate.'})

    if (age > 50 or age < 25) or (blood_pressure > 150 and heart_rate > 100):
        prediction_text = "<b>High risk of heart disease</b>"
    else:
        prediction = model.predict([[age, gender, blood_pressure, heart_rate]])[0]
        prediction_text = "<b>High risk of heart disease</b>" if prediction == 1 else "<b>Low risk of heart disease</b>"

    return prediction_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
