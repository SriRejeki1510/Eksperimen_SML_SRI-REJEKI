import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv("wine_quality_preprocessing/wine_quality_preprocessed.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aktifkan autolog
mlflow.sklearn.autolog()

# Training
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
