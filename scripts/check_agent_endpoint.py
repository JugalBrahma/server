"""Send one sample request to the agent endpoint."""

import os

import requests

API_URL = os.getenv("API_URL", "http://10.0.2.2:8000") + "/agent"
FIREBASE_TOKEN = os.getenv("FIREBASE_TOKEN", "")
FOLDER_ID = os.getenv("FOLDER_ID")


def main() -> None:
    if not FIREBASE_TOKEN:
        print("Set FIREBASE_TOKEN before running this script.")
        return

    payload = {"query": "What is machine learning?"}
    if FOLDER_ID:
        payload["folder_id"] = FOLDER_ID

    headers = {"Authorization": f"Bearer {FIREBASE_TOKEN}"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    except requests.RequestException as error:
        print(f"Request failed: {error}")
        return

    print(f"Status code: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
