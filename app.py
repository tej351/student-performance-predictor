from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "Student Performance Predictor API"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array([[
        data["StudyHours"],
        data["Attendance"],
        data["PreviousMarks"],
        data["Assignments"]
    ]])

    prediction = model.predict(features)

    return jsonify({
        "Predicted Marks": round(prediction[0], 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
