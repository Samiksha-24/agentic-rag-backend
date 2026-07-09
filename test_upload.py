import requests

url = "http://127.0.0.1:8000/upload"

files = [
    ("files", open(r"D:\SAMIKSHA_UPDATED_RESUME_FINAL.pdf", "rb"))
]

response = requests.post(url, files=files)

print("Status:", response.status_code)
print(response.text)