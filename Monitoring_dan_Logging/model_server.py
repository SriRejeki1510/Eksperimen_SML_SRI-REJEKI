# model_server.py

from fastapi import FastAPI, Request
import joblib
import pandas as pd
import os
from prometheus_client import start_http_server, Gauge
import time
import threading

app = FastAPI()

# Load model saat start
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "Membangun_model", "model", "model.pkl")
model = joblib.load(model_path)

# Prometheus metrics
inference_requests = Gauge("inference_requests_total", "Total jumlah permintaan inferensi")
inference_latency = Gauge("inference_latency_seconds", "Waktu latensi inferensi dalam detik")
model_accuracy = Gauge("model_accuracy_score", "Akurasi model")

# Jalankan Prometheus metrics server di thread terpisah
def run_metrics():
    start_http_server(8000)
    while True:
        model_accuracy.set(0.87)  # nilai simulasi, bisa ubah dari hasil validasi sebenarnya
        time.sleep(10)

threading.Thread(target=run_metrics, daemon=True).start()

@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()
    data = await request.json()
    df = pd.DataFrame(data["data"])
    pred = model.predict(df).tolist()
    end_time = time.time()

    # Update metrics
    inference_requests.inc()
    inference_latency.set(round(end_time - start_time, 4))

    return {"prediksi": pred}
