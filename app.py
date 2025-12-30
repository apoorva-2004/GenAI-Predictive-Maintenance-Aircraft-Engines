from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("predictive_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Predictive Maintenance Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    data = np.array(data).reshape(1, -1)
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    return jsonify({
        "Prediction": "Failure" if prediction == 1 else "No Failure"
    })

if __name__ == "__main__":
    app.run(debug=True)
