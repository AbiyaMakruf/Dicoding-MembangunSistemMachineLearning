from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# === METRIK API MODEL ===
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  # Total request masuk
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  # Latensi request
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  # Throughput
ERROR_COUNT = Counter('http_requests_error_total', 'Total number of failed requests')  # Jumlah error
SUCCESS_COUNT = Counter('http_requests_success_total', 'Total number of successful requests')  # Jumlah sukses
REQUEST_SIZE = Histogram('http_request_size_bytes', 'Size of HTTP request payload (bytes)')  # Ukuran request
RESPONSE_SIZE = Histogram('http_response_size_bytes', 'Size of HTTP response payload (bytes)')  # Ukuran response

# === METRIK SISTEM ===
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')
DISK_USAGE = Gauge('system_disk_usage', 'Disk Usage Percentage')
NET_BYTES_SENT = Gauge('system_net_bytes_sent', 'Total Bytes Sent')
NET_BYTES_RECV = Gauge('system_net_bytes_recv', 'Total Bytes Received')

# Endpoint untuk Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    # Update metrik sistem setiap kali /metrics diakses
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    net_io = psutil.net_io_counters()
    NET_BYTES_SENT.set(net_io.bytes_sent)
    NET_BYTES_RECV.set(net_io.bytes_recv)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint untuk model
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    data = request.get_json()

    # Catat ukuran request
    if data:
        REQUEST_SIZE.observe(len(str(data).encode('utf-8')))

    api_url = "http://127.0.0.1:5005/invocations"

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time

        REQUEST_LATENCY.observe(duration)
        SUCCESS_COUNT.inc()

        # Catat ukuran response
        RESPONSE_SIZE.observe(len(response.content))

        return jsonify(response.json())

    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
