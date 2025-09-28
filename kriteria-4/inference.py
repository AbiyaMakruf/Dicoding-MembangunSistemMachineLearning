import requests
import json

# URL endpoint Flask wrapper (bukan langsung ke MLflow)
PREDICT_URL = "http://127.0.0.1:8000/predict"

sample_input = {
    "instances": [
        [0.5869565217391305, 0.24791498520312072, 0.4, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.8, 0.35, 0.6, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    ]
}

def predict(input_data):
    headers = {"Content-Type": "application/json"}
    response = requests.post(PREDICT_URL, headers=headers, data=json.dumps(input_data))

    if response.status_code == 200:
        print("✅ Prediction:", response.json())
    else:
        print("❌ Error:", response.status_code, response.text)

if __name__ == "__main__":
    predict(sample_input)
