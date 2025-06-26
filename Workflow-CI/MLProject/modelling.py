import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set MLflow Tracking ke lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine Quality Sri Rejeki")

# Load data hasil preprocessing
df = pd.read_csv("wine_quality_preprocessing/wine_quality_preprocessed.csv")

# Cek kolom yang tersedia (debugging)
print("Kolom-kolom:", df.columns)

# Pisahkan fitur dan target
X = df.drop("label", axis=1)
y = df["label"]

# Split data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Training model
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Akurasi:", acc)
