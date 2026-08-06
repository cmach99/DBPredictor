
import requests
import json

# Test the local API
url = "http://localhost:8000/predict"

# Replace with actual feature values matching your dataset
# This is just an example — use /feature-info to see the exact order
payload = {
    "features": [45.0, 4.7, 75.0, 6.5, 4.2, 1.8, 1.2, 2.5, 0.8, 28.5, 1.0]
}

response = requests.post(url, json=payload)
print("Status:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))
