from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Collect input values from the form
    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["bp"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["ecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    # Dataset feature names
    feature_names = [
        "age",
        "sex",
        "chest pain type",
        "resting blood pressure",
        "serum cholestoral",
        "fasting blood sugar",
        "resting electrocardiographic results",
        "max heart rate",
        "exercise induced angina",
        "oldpeak",
        "ST segment",
        "major vessels",
        "thal"
    ]

    # Create DataFrame
    input_data = pd.DataFrame(
        [features],
        columns=feature_names
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Convert prediction to result
    if prediction == 1:
        result = "⚠️ Heart Disease Detected"
        status = "danger"
    else:
        result = "✅ No Heart Disease Detected"
        status = "success"

    return render_template(
        "result.html",
        prediction=result,
        status=status
    )


if __name__ == "__main__":
    app.run(debug=True)