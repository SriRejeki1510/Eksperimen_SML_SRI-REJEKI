import joblib
import pandas as pd
import os

# Path model dan data
base_dir = os.path.dirname(os.path.abspath(__file__))

# Sesuaikan nama file model kalau kamu rename
model_path = os.path.join(base_dir, "..", "Membangun_model", "model", "model.pkl")
data_path = os.path.join(base_dir, "..", "Membangun_model", "wine_quality_preprocessing", "wine_quality_preprocessed.csv")

# Load model
model = joblib.load(model_path)

# Load data
df = pd.read_csv(data_path)

# Ambil 5 sampel awal (tanpa kolom label)
sample = df.drop("label", axis=1).iloc[:5]

# Prediksi
prediksi = model.predict(sample)

# Tampilkan hasil
print("Input Data:")
print(sample)
print("\nHasil Prediksi:")
print(prediksi)
