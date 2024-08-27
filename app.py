from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the KNN model and the scaler from the pickle files
with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input features from the form
        features = [float(request.form['feature1']),
                    float(request.form['feature2']),
                    float(request.form['feature3']),
                    float(request.form['feature4'])]

        # If you have default or placeholder values for the missing features
        # you can add them here, for example:
        missing_features = [0, 0, 0]  # Placeholder values for missing features
        features.extend(missing_features)

        # Reshape and scale the features
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = knn_model.predict(scaled_features)

        # Example: assuming binary classification, with labels 0 and 1
        result = 'Class 1' if prediction[0] == 1 else 'Class 0'

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
