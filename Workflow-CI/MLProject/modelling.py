import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

mlflow.set_experiment("Wine Quality Sri Rejeki")

df = pd.read_csv("wine_quality_preprocessing/wine_quality_preprocessed.csv", sep=',')

print("Kolom-kolom:", df.columns)

# Tangani NaN di fitur (X) - isi dengan rata-rata
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if col != 'label' and df[col].isnull().any(): # Jangan isi NaN di label dengan rata-rata!
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"Mengisi NaN di kolom '{col}' dengan rata-rata ({mean_val})")

# === PERUBAHAN KRUSIAL UNTUK KOLOM 'label' (TARGET) ===
# 1. Pastikan kolom 'label' adalah INTEGER (kategori diskrit)
# 2. Tangani NaN di kolom 'label' dengan menghapus barisnya (paling aman untuk klasifikasi)
initial_rows = len(df)
# Hapus baris dengan NaN di kolom 'label'
df.dropna(subset=['label'], inplace=True)
rows_after_dropna = len(df)
if initial_rows != rows_after_dropna:
    print(f"Menghapus {initial_rows - rows_after_dropna} baris karena nilai NaN di kolom 'label'.")

# Konversi kolom 'label' ke tipe integer. Ini memastikan ia diperlakukan sebagai kategori diskrit.
# Jika ada nilai float di 'label' (misal 5.0, 6.0), ini akan mengubahnya menjadi 5, 6.
# Jika ada nilai float non-integer (misal 5.5), ini akan error atau dibulatkan,
# jadi pastikan label Anda memang integer.
df['label'] = df['label'].astype(int) 
print(f"Kolom 'label' dikonversi ke integer. Contoh nilai label: {df['label'].unique()}")
# === AKHIR PERUBAHAN KRUSIAL ===

# Pisahkan fitur dan target
X = df.drop("label", axis=1)
y = df["label"]

# Split data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Training model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi:", acc)

print("MLflow run should now complete successfully!")