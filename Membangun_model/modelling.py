import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi MLflow lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine Quality Sri Rejeki")

# Load dataset hasil preprocessing (cek nama kolom target yang valid)
df = pd.read_csv("wine_quality_preprocessing/wine_quality_preprocessed.csv")

# Tampilkan nama-nama kolom (debugging, bisa dihapus nanti)
print("Kolom-kolom:", df.columns)

# Pastikan nama kolom target sesuai
# Gantilah 'quality' sesuai dengan kolom target sebenarnya di dataset-mu
target_column = "quality"

# Pisahkan fitur dan target
X = df.drop(columns=[target_column])
y = df[target_column]

# Bagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Aktifkan autologging MLflow
mlflow.sklearn.autolog()

# Training model
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.4f}")
