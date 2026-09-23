from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["StudyHours"])
    attendance = float(request.form["Attendance"])
    previous_marks = float(request.form["PreviousMarks"])
    assignments = float(request.form["Assignments"])

    features = np.array([[
        study_hours,
        attendance,
        previous_marks,
        assignments
    ]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
