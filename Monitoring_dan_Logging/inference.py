# inference.py

import pandas as pd
import requests
import os

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "Membangun_model", "wine_quality_preprocessing", "wine_quality_preprocessed.csv")
df = pd.read_csv(data_path)
sample = df.drop("label", axis=1).iloc[:5]

# Kirim data ke model API
response = requests.post("http://127.0.0.1:8000/predict", json={"data": sample.to_dict(orient="records")})

# Tampilkan hasil
print("Input Data:")
print(sample)
print("\nHasil Prediksi:")
print(response.json())
