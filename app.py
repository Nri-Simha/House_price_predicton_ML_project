# app.py

from flask import Flask, request, jsonify, render_template # type: ignore
import pickle
import numpy as np # type: ignore

# Load the trained model
model_path = 'house_predictor_model.pickle'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
   

    return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
