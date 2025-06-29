import setuptools  # Pastikan setuptools diimpor dulu untuk menghindari konflik distutils
import mlflow
import mlflow.sklearn

# Aktifkan autologging sebelum import sklearn lainnya
mlflow.sklearn.autolog()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set experiment MLflow
mlflow.set_experiment("Wine Quality Sri Rejeki")

# Load data
df = pd.read_csv("wine_quality_preprocessing/wine_quality_preprocessed.csv", sep=',')

print("Kolom-kolom:", df.columns)

# Tangani NaN di fitur (X) - isi dengan rata-rata
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if col != 'label' and df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"Mengisi NaN di kolom '{col}' dengan rata-rata ({mean_val})")

# Tangani NaN di label dengan menghapus baris
initial_rows = len(df)
df.dropna(subset=['label'], inplace=True)
rows_after_dropna = len(df)
if initial_rows != rows_after_dropna:
    print(f"Menghapus {initial_rows - rows_after_dropna} baris karena nilai NaN di kolom 'label'.")

# Konversi label ke integer
df['label'] = df['label'].astype(int)
print(f"Kolom 'label' dikonversi ke integer. Contoh nilai label: {df['label'].unique()}")

# Pisahkan fitur dan target
X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gunakan context manager mlflow.start_run() untuk kontrol run
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Akurasi:", acc)

    # Log metric manual jika perlu (autolog juga sudah otomatis log metric)
    mlflow.log_metric("accuracy", acc)

print("MLflow run should now complete successfully!")
