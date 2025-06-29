import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np # Tambahkan import numpy jika belum ada

# MLflow run dari command line sudah memulai run secara otomatis
mlflow.set_experiment("Wine Quality Sri Rejeki")

# Load data hasil preprocessing (pastikan pakai sep=',')
df = pd.read_csv("wine_quality_preprocessing/wine_quality_preprocessed.csv", sep=',')

# Cek kolom yang tersedia (debugging)
print("Kolom-kolom:", df.columns)

# === TAMBAHKAN KODE INI UNTUK MENANGANI NaN ===
# 1. Tangani NaN di fitur (X) - isi dengan rata-rata
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']: # Hanya untuk kolom numerik
        df[col] = df[col].replace([np.inf, -np.inf], np.nan) # Ganti inf/-inf dengan NaN jika ada
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"Mengisi NaN di kolom '{col}' dengan rata-rata ({mean_val})")

# 2. Tangani NaN di target (y) - cara paling aman adalah menghapus barisnya
# Sebelum split, hapus baris dengan NaN di kolom 'label' (target)
initial_rows = len(df)
df.dropna(subset=['label'], inplace=True)
rows_after_dropna = len(df)
if initial_rows != rows_after_dropna:
    print(f"Menghapus {initial_rows - rows_after_dropna} baris karena nilai NaN di kolom 'label'.")

# === AKHIR KODE PENANGANAN NaN ===


# Pisahkan fitur dan target
X = df.drop("label", axis=1)
y = df["label"] # Pastikan kolom 'label' tidak ada NaN lagi


# Split data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Training model (Sekarang ini akan dijalankan dalam run yang sudah dimulai oleh mlflow run .)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi:", acc)

print("MLflow run should now complete successfully!")