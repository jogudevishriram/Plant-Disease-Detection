import requests

url = "http://localhost:5000/predict"
files = {"image": open("apple.jpeg", "rb")}  # Replace with your test image
response = requests.post(url, files=files)
print(response.json())