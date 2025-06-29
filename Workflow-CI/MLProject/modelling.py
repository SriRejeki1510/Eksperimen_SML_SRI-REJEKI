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

df = pd.read_csv("wine_quality_preprocessing/wine_quality_preprocessed.csv", sep=',')

print("Kolom-kolom:", df.columns)

for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if col != 'label' and df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"Mengisi NaN di kolom '{col}' dengan rata-rata ({mean_val})")

initial_rows = len(df)
df.dropna(subset=['label'], inplace=True)
rows_after_dropna = len(df)
if initial_rows != rows_after_dropna:
    print(f"Menghapus {initial_rows - rows_after_dropna} baris karena nilai NaN di kolom 'label'.")

df['label'] = df['label'].astype(int)
print(f"Kolom 'label' dikonversi ke integer. Contoh nilai label: {df['label'].unique()}")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi:", acc)

# Jika ingin log metric manual juga, bisa dipanggil
mlflow.log_metric("accuracy", acc)

print("MLflow run should now complete successfully!")
