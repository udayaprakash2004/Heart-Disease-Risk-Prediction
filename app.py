from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load saved model
model = pickle.load(open("heart_model.pkl", "rb"))

@app.route("/")
def home():
    # Page 1: Input form
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs
    features = [
        int(request.form["age"]),
        int(request.form["sex"]),
        int(request.form["cp"]),
        int(request.form["bp"]),
        int(request.form["chol"]),
        int(request.form["fbs"]),
        int(request.form["ecg"]),
        int(request.form["thalach"]),
        int(request.form["exang"]),
        float(request.form["oldpeak"]),
        int(request.form["slope"]),
        int(request.form["ca"]),
        int(request.form["thal"])
    ]

    # Convert to DataFrame (important)
    input_data = pd.DataFrame([features], columns=model.feature_names_in_)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "⚠️ Heart Disease Detected"
        status = "danger"
    else:
        result = "✅ No Heart Disease Detected"
        status = "success"

    # Page 2: Result page
    return render_template("result.html", prediction=result, status=status)

if __name__ == "__main__":
    app.run(debug=True)
