import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Get base directory of script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset using a relative path
data_path = os.path.join(BASE_DIR, "..", "data", "insurance.csv")
data = pd.read_csv(data_path)

#  (OHE)
data = pd.get_dummies(data, drop_first=True)

# features
X = data.drop("charges", axis=1)
y = data["charges"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

#  plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")

# Save plot using a relative path
reports_path = os.path.join(BASE_DIR, "..", "reports", "actual_vs_predicted.png")
plt.savefig(reports_path)
plt.show()
