# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Import Required Libraries

Step 2: Load the Dataset

Step 3: Copy Data & Drop Unwanted Columns

Step 4: Check Data Quality

Step 5: Encode Categorical Variables

Step 6: Define Features (X) and Target (y)

Step 7: Split into Training and Testing Sets

Step 8: Build and Train Logistic Regression Model

Step 9: Make Predictions

Step 10: Evaluate the Model

Step 11: Predict for a New Student

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sajin Praneeth S R
RegisterNumber: 212224060229

*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")
print("First 5 rows of the dataset:")
print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())

X = data1.iloc[:, :-1]
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])
```

## Output:

<img width="757" height="335" alt="image" src="https://github.com/user-attachments/assets/3b70c0e8-bce6-49a1-a86c-d3c32a386392" />

<img width="731" height="326" alt="image" src="https://github.com/user-attachments/assets/255d4094-a544-419a-98f2-db713d41363b" />

<img width="472" height="406" alt="image" src="https://github.com/user-attachments/assets/6b2f43d8-b704-461c-a214-3f02aab21fdb" />

<img width="857" height="335" alt="image" src="https://github.com/user-attachments/assets/0458e416-19f0-4cbb-9673-a06d3f0070e6" />

<img width="835" height="477" alt="image" src="https://github.com/user-attachments/assets/a6e05f8d-b1e5-477c-923c-9384f23e48c0" />

<img width="877" height="87" alt="image" src="https://github.com/user-attachments/assets/173fd6b4-2722-4e4b-9147-8b62b5590cc0" />

<img width="626" height="277" alt="image" src="https://github.com/user-attachments/assets/310ce5b6-8949-48ae-951e-78fcf16f27a4" />

<img width="597" height="52" alt="image" src="https://github.com/user-attachments/assets/e2a411c4-6ccd-4821-8b29-7d4610f15cd0" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
