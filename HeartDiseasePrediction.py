import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# 1. Load dataset
data = pd.read_csv("dataset_heart.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("Dataset columns:")
print(data.columns.tolist())

print("\nHeart disease values:")
print(data["heart disease"].value_counts())


# 2. Prepare features and target

# Convert target:
# 1 = No Heart Disease
# 2 = Heart Disease
#
# Convert to:
# 0 = No Heart Disease
# 1 = Heart Disease

data["heart disease"] = data["heart disease"].map({
    1: 0,
    2: 1
})

X = data.drop("heart disease", axis=1)
y = data["heart disease"]


# 3. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Machine learning models

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ])
}


# 5. Train and evaluate models

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        )
    }

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# 6. Select best model using F1 score

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = results[best_model_name]["model"]


print("\n" + "=" * 50)
print("BEST MODEL")
print("=" * 50)

print(best_model_name)

print(
    f"Best F1 Score: "
    f"{results[best_model_name]['f1']:.4f}"
)


# 7. Save best model

with open("heart_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest model saved successfully as heart_model.pkl")