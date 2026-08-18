import pandas as p  # for loading and extracting the info from the dataset
from sklearn.linear_model import LogisticRegression  # this is the model used for the prediction
from sklearn.model_selection import train_test_split  # for splitting the dataset
from sklearn.metrics import accuracy_score, confusion_matrix  # these are metrics for checking the accuracy of a model
import pickle  # for saving the trained model


# Step 1: Load the dataset

dataset = p.read_csv('dataset_heart.csv')
# dataset fetched successfully


# Step 2: Separate input features and output target


# Input features (first 13 columns)
input = dataset.iloc[:, :13]

# Output / Target column
# 1 → No heart disease
# 2 → Has heart disease
output = dataset['heart disease']

# Convert target values to binary (recommended for Logistic Regression)
# 0 → No heart disease
# 1 → Has heart disease
output = output.map({1: 0, 2: 1})


# Step 3: Train-test split

x_train, x_test, y_train, y_test = train_test_split(
    input,
    output,
    random_state=42,
    test_size=0.2
)

# Split the dataset into training and testing sets
# x_train, y_train → data used to train the model
# x_test, y_test   → data used to test/evaluate the model
# input  → all features (columns except target)
# output → target column (here 'heart disease')
# test_size=0.2 → 20% of data will be used for testing, 80% for training
# random_state=42 → ensures the split is always the same every time you run
# (so results are reproducible; you will get same x_train/x_test each run)
# and also you will get the same accuracy


# Step 4: Create the Logistic Regression model

model = LogisticRegression(
    max_iter=1000
)  # max_iter gives the optimization algorithm more steps
# to find the best coefficients (weights) for the features

# model constructed


# Step 5: Train the model

model.fit(x_train, y_train)  # training the model


# Step 6: Predict on test data

y_pred = model.predict(x_test)


# Step 7: Predict for a new person (manual input)


# Attribute explanation for the new person:
# age       → Age of the person in years
# sex       → 1 = Male, 0 = Female
# cp        → Chest pain type (1: Typical angina, 2: Atypical angina,
#             3: Non-anginal pain, 4: Asymptomatic)
# trestbps → Resting blood pressure (in mm Hg)
# chol      → Serum cholesterol (mg/dl)
# fbs       → Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
# restecg  → Resting ECG results
#             0 = Normal
#             1 = ST-T wave abnormality
#             2 = Left ventricular hypertrophy
# thalach  → Maximum heart rate achieved
# exang    → Exercise induced angina (1 = Yes, 0 = No)
# oldpeak  → ST depression induced by exercise relative to rest
# slope    → Slope of the peak exercise ST segment
#             1 = Upsloping
#             2 = Flat
#             3 = Downsloping
# ca       → Number of major vessels (0–3) colored by fluoroscopy
# thal     → Thalassemia
#             3 = Normal
#             6 = Fixed defect
#             7 = Reversible defect

# You need to create a 2D array with these values
# (because scikit-learn always expects 2D input)
new_person = p.DataFrame(
    [[55, 1, 2, 140, 240, 0, 1, 150, 0, 1.5, 2, 0, 3]],
    columns=x_train.columns
)  # we need to give with the column names

prediction = model.predict(new_person)

print(f'Predicted: {prediction}')

# Step 8: Model Evaluation
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy will be: {acc}')

# Confusion Matrix (optional but useful)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Step 9: Save the trained model (for website use)
pickle.dump(model, open("heart_model.pkl", "wb"))
print("Model saved successfully as heart_model.pkl")

