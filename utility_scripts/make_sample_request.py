import requests
import base64

with open("/home/shubham/Downloads/Documents/CR_copy-EN.pdf", "rb") as f:
    pdf = base64.b64encode(f.read()).decode()

response = requests.post("http://0.0.0.0:9000/json/qna/pdf", json={"pdf": pdf, "questions": ["What is the name of the company?"]})
print(response.json())