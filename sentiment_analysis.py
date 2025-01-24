import requests
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables from the .env file
api_key = os.getenv('API_KEY')
model_path=os.getenv('MODEL_PATH')
prod_path=os.getenv('PROD_PATH')
obj_path=os.getenv('OBJ_PATH')
#API key
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
API_KEY = api_key  

headers = {"Authorization": f"Bearer {API_KEY}"}


def analyze_sentiment(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]  
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return {"label": "ERROR", "score": 0.0}


transcription = "The product is amazing and I'm really happy with it."
sentiment = analyze_sentiment(transcription)
print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']}")
