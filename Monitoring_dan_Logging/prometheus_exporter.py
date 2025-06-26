from prometheus_client import start_http_server, Gauge
import random
import time

# Metrik monitoring
inference_requests = Gauge("inference_requests_total", "Total jumlah permintaan inferensi")
inference_latency = Gauge("inference_latency_seconds", "Waktu latensi inferensi dalam detik")
model_accuracy = Gauge("model_accuracy_score", "Akurasi model")

# Menjalankan server Prometheus exporter di port 8000
start_http_server(8000)

# Simulasi update metriks setiap 5 detik
while True:
    # Tambahkan permintaan inferensi secara acak
    inference_requests.inc(random.randint(1, 3))

    # Set latensi inferensi secara acak
    inference_latency.set(round(random.uniform(0.01, 0.5), 3))

    # Set akurasi model (simulasi)
    model_accuracy.set(round(random.uniform(0.80, 0.95), 3))

    # Tunggu 5 detik sebelum update berikutnya
    time.sleep(5)
