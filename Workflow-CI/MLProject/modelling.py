import setuptools
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Wine Quality Sri Rejeki")

# Load dan preprocess data sesuai kode Anda
df = pd.read_csv("wine_quality_preprocessing/wine_quality_preprocessed.csv", sep=',')

# ... (kode preprocessing dan training model)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi:", acc)

mlflow.log_metric("accuracy", acc)
print("MLflow run should now complete successfully!")
