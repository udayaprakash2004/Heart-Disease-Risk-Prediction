# ❤️ Heart Disease Prediction ML

### An end-to-end machine learning project that predicts the **presence or absence of heart disease** using Logistic Regression and Flask, with a responsive CSS frontend.

---

## 🚀 Live Demo

Try it online: [Try Online](https://heart-disease-prediction-using-ml-fm2i.onrender.com/)

---

## 🛠 Technologies Used

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn (Logistic Regression)
* **Web Framework:** Flask
* **Frontend:** HTML, CSS
* **Deployment:** Render (Free hosting)
* **Data:** Heart disease clinical dataset (`dataset_heart.csv`)

---

## 💡 Features

* Predicts heart disease based on clinical attributes  
* Clean and responsive web interface  
* Hosted online for real-time usage  
* Uses a trained Logistic Regression model for accurate predictions  
* Easy-to-use input form and result display

---

## 📂 Project Structure

```
Heart-Disease-Prediction-Using-ML/
├── app.py 📝 Flask backend
├── heart_model.pkl 📦 Trained ML model
├── requirements.txt 📄 Python dependencies
├── runtime.txt ⚙️ Python version for Render
├── templates/ 📁 HTML templates
│ ├── index.html 🖥️ Input form
│ └── result.html 🖥️ Prediction result
├── static/ 📁 Static assets
│ └── style.css 🎨 CSS styling
└── dataset_heart.csv 📊 Sample dataset
```

---

## ⚡ Installation & Local Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/udayaprakash2004/Heart-Disease-Risk-Prediction
```

### 2️⃣ Navigate into the project folder

```bash
cd Heart-Disease-Prediction-Using-ML
```

### 3️⃣ Create & activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Flask application

```bash
python app.py
```

### 6️⃣ Open your browser

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

1. User enters patient data (age, sex, blood pressure, etc.)
2. Flask loads the trained Logistic Regression model
3. Input data is processed and passed to the model
4. The model predicts **Heart Disease Detected** or **No Heart Disease**
5. Result is displayed on the webpage

---

## 🚀 Future Improvements

* Add more ML algorithms for better accuracy
* Include confidence scores in predictions
* Enhance frontend with interactive charts
* Store historical predictions for analysis

---

## 📫 Contact

* **GitHub:** [https://github.com/udayaprakash2004](https://github.com/udayaprakash2004)
* **LinkedIn:** [https://www.linkedin.com/in/udayaprakash12](https://www.linkedin.com/in/udayaprakash12)


---

## 📄 License

This project is licensed under the **MIT License**.

```
