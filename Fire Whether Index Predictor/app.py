from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("ridge.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input features from the form
         # 1. Read inputs from form (use exact form field names)
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['WIND'])   # WIND from HTML = Ws in model
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        DC = float(request.form['DC'])
        ISI = float(request.form['ISI'])
        BUI = float(request.form['BUI'])

        # 2. Arrange them in the correct model training order
        # IMPORTANT: This order MUST match your training feature order
        # ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']
        inputs_array = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI]])

        # Convert to numpy array and reshape for model
        #inputs_array = np.array(inputs).reshape(1, -1)

        # Scale the input data
        scaled_inputs = scaler.transform(inputs_array)

        # Make prediction
        prediction = model.predict(scaled_inputs)[0]

        # Return result page
        prediction = max(0.0, prediction)

        return render_template('home.html', prediction=round(prediction, 4))

if __name__ == '__main__':
    app.run(debug=True)