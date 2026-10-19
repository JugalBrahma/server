"""Upload a local PDF to the API."""

import os

import requests

API_URL = os.getenv("API_URL", "http://10.0.2.2:8000") + "/upload-pdf"
FIREBASE_TOKEN = os.getenv("FIREBASE_TOKEN", "")
FOLDER_ID = os.getenv("FOLDER_ID")
PDF_PATH = os.getenv("PDF_PATH", "")


def main() -> None:
    if not FIREBASE_TOKEN:
        print("Set FIREBASE_TOKEN before running this script.")
        return
    if not PDF_PATH or not os.path.isfile(PDF_PATH):
        print("Set PDF_PATH to an existing PDF file.")
        return

    headers = {"Authorization": f"Bearer {FIREBASE_TOKEN}"}
    if FOLDER_ID:
        headers["folder_id"] = FOLDER_ID

    try:
        with open(PDF_PATH, "rb") as pdf_file:
            files = {"file": (os.path.basename(PDF_PATH), pdf_file, "application/pdf")}
            response = requests.post(API_URL, headers=headers, files=files, timeout=60)
    except requests.RequestException as error:
        print(f"Request failed: {error}")
        return

    print(f"Status code: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
