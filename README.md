# ❤️ Heart Disease Risk Prediction


### An end-to-end machine learning project that predicts the presence or absence of heart disease using multiple machine learning algorithms and a Flask web application.


---


## 📌 Project Overview


This project is a machine learning-based web application for predicting heart disease from clinical input parameters.


Multiple machine learning algorithms are evaluated, and the model with the best F1 score is selected automatically.


The selected model is integrated with a Flask web application where users can enter patient information and receive a prediction.


> **Project Note:** This repository is a modified and extended version of an existing MIT-licensed project. The project has been adapted with updated model training, preprocessing, Flask integration, and application improvements.


---


## 🧠 Machine Learning Models


The project evaluates the following algorithms:


- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)


The model with the highest F1 score is automatically selected and saved as:


```text
heart_model.pkl
📊 Model Performance
Best Model: Logistic Regression
Metric	Score
Accuracy	85.19%
Precision	78.57%
Recall	91.67%
F1 Score	84.62%
Model Comparison
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	85.19%	78.57%	91.67%	84.62%
Random Forest	83.33%	80.00%	83.33%	81.63%
SVM	81.48%	76.92%	83.33%	80.00%
Decision Tree	79.63%	74.07%	83.33%	78.43%
KNN	79.63%	72.41%	87.50%	79.25%
🛠 Technologies Used
Programming Language: Python
Machine Learning: Scikit-learn
Web Framework: Flask
Data Processing: Pandas, NumPy
Frontend: HTML, CSS
Model Storage: Pickle
Version Control: Git and GitHub
✨ Features
Heart disease prediction using machine learning
Multiple machine learning algorithms
Automatic best-model selection
Flask-based web application
Responsive web interface
Clinical input form
Prediction result page
Trained model saved as heart_model.pkl
📂 Project Structure
Heart-Disease-Risk-Prediction/
│
├── app.py
├── HeartDiseasePrediction.py
├── heart_model.pkl
├── dataset_heart.csv
├── requirements.txt
├── runtime.txt
├── .python-version
├── LICENSE
├── README.md
│
├── static/
│   └── style.css
│
└── templates/
    ├── index.html
    └── result.html
⚡ Installation & Local Setup
1. Clone the repository
git clone https://github.com/udayaprakash2004/Heart-Disease-Risk-Prediction.git
2. Navigate to the project folder
cd Heart-Disease-Risk-Prediction
3. Create a virtual environment
python -m venv .venv
4. Activate the virtual environment
Windows PowerShell
.\.venv\Scripts\Activate.ps1

If PowerShell blocks the activation script:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then:

.\.venv\Scripts\Activate.ps1
5. Install dependencies
pip install -r requirements.txt
6. Train the machine learning models
python HeartDiseasePrediction.py

This evaluates the machine learning algorithms and saves the best model as:

heart_model.pkl
7. Start the Flask application
python app.py
8. Open the application

Open your browser and visit:

http://127.0.0.1:5000
🔄 How the System Works
User
  ↓
Enter Clinical Information
  ↓
Flask Web Application
  ↓
Input Data Processing
  ↓
Trained Machine Learning Model
  ↓
Heart Disease Prediction
  ↓
Result Displayed
📋 Input Parameters

The application uses clinical parameters including:

Age
Sex
Chest pain type
Resting blood pressure
Serum cholesterol
Fasting blood sugar
Resting electrocardiographic results
Maximum heart rate
Exercise-induced angina
Oldpeak
ST segment
Major vessels
Thalassemia
🚀 Live Demo

Live demo: Coming soon

A live demo link will be added after deploying this version of the project.

🔮 Future Improvements
Add prediction confidence/probability
Improve the user interface
Add visualization of model performance
Add confusion matrix and classification reports
Add additional machine learning algorithms
Deploy the application online
Improve input validation and error handling
👨‍💻 Author

Udaya Prakash

GitHub:

https://github.com/udayaprakash2004

Project Repository:

https://github.com/udayaprakash2004/Heart-Disease-Risk-Prediction

📄 License

This project contains an MIT License.