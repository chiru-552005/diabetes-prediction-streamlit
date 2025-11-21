import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# Features and target
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

# Evaluate
preds = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(Y_test, preds))

# Save model and scaler
pickle.dump(model, open("diabetes_model.sav", "wb"))
pickle.dump(scaler, open("scaler.sav", "wb"))
print("Saved diabetes_model.sav and scaler.sav in current directory.")