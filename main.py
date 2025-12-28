# loading_dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# data_loading
print("\n" + "=" * 50)
print("DATA LOADING")
print("=" * 50)

df = pd.read_csv("../data/diabetes.csv")
print(f"Dataset shape : {df.shape}")
print(f"\nFirst Five rows :")
print(df.head())

print("\nGenerating correlation heatmap...")
correlation = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# data cleaning
print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

columns_to_check = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]
print("\nZeroes before cleaning :")
for col in columns_to_check:
    zeros = (df[col] == 0).sum()
    print(f"{col} : {zeros} zeros")

df = df.drop("Insulin", axis=1)
remaining = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]
for col in remaining:
    median = df[df[col] != 0][col].median()
    df[col].replace(0, median, inplace=True)

print("\nZeroes after cleaning : ")
for col in remaining:
    zeros = (df[col] == 0).sum()
    print(f"{col} : {zeros} zeros")

# Data preparation
print("\n" + "=" * 50)
print("DATA PREPARATION")
print("=" * 50)

x = df.drop("Outcome", axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f"Training set : {x_train.shape[0]} samples, {x_train.shape[1]} Features")
print(f"Testing set : {x_test.shape[0]} samples, {x_test.shape[1]} Features")

# training model
print("\n" + "=" * 50)
print("TRAINING MODEL...")
print("=" * 50)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaler, y_train)
y_predict = model.predict(x_test_scaler)
print("MODEL TRAINING COMPLETED!")

# result
print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

acc = accuracy_score(y_test, y_predict)
print(f"Accuracy : {acc * 100:.2f}%")
report = classification_report(y_test, y_predict)
print(f"classification_report :\n{report}")
cl_matrix = confusion_matrix(y_test, y_predict)
print(f"confusion_matrix :\n{cl_matrix}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cl_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"✓ Dataset: {df.shape[0]} patients, {df.shape[1] - 1} features")
print(f"✓ Model: Logistic Regression with StandardScaler")
print(f"✓ Accuracy: {acc * 100:.2f}%")
print(f"✓ Key Finding: Glucose (0.47) and BMI (0.29) are strongest predictors")
print(f"✓ Weakness: Model missed {21} diabetic cases (false negatives)")
print(f"✓ Improvement: Could try Random Forest or tune decision threshold")
