import os
import sys
import vosk
import pandas as pd
import pyaudio
import requests
import gspread
import json
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagTokenForGeneration
from transformers import DPRQuestionEncoderTokenizerFast, BartTokenizerFast
import faiss
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# File paths from .env
PRODUCT_DATA_FILE = os.getenv("PRODUCT_DATA_FILE")
OBJECTIONS_DATA_FILE = os.getenv("OBJECTIONS_DATA_FILE")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE")
SHEET_NAME = os.getenv("SHEET_NAME")

# Hugging Face API details from .env
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Load product and objection data
print("Loading product data...")
product_data = pd.read_csv(PRODUCT_DATA_FILE)
product_descriptions = product_data['description'].tolist()
product_titles = product_data['title'].tolist()

print("Loading objections data...")
objections_data = pd.read_csv(OBJECTIONS_DATA_FILE)
objections = objections_data['objection'].tolist()
responses = objections_data['response'].tolist()

# Load Sentence Transformer model
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for product descriptions and objections
print("Generating embeddings...")
product_embeddings = model.encode(product_descriptions)
objection_embeddings = model.encode(objections)

# Create FAISS indices for products and objections
print("Creating FAISS indices...")
product_index = faiss.IndexFlatL2(product_embeddings.shape[1])
product_index.add(product_embeddings)

objection_index = faiss.IndexFlatL2(objection_embeddings.shape[1])
objection_index.add(objection_embeddings)

# Initialize Vosk Model
print("Loading Vosk model...")
vosk_model = vosk.Model(VOSK_MODEL_PATH)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

# Initialize pyaudio
print("Initializing audio stream...")
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)

def analyze_sentiment(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        elif isinstance(result, dict):
            return result
    return {"label": "ERROR", "score": 0.0}

# Google Sheets API Setup
print("Setting up Google Sheets API...")
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
client = gspread.authorize(creds)
sheet = client.open(SHEET_NAME).sheet1

def append_to_sheet(sentiment, transcription):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment_label = sentiment.get('label', 'Unknown')
    sentiment_score = sentiment.get('score', 0.0)
    sheet.append_row([timestamp, sentiment_label, sentiment_score, transcription])

# Function for product recommendation
def recommend_products(query):
    query_embedding = model.encode([query])
    distances, indices = product_index.search(query_embedding, 3)
    recommendations = [(product_titles[i], product_descriptions[i]) for i in indices[0]]
    return recommendations

# Function for objection handling
def handle_objection(query):
    query_embedding = model.encode([query])
    distances, indices = objection_index.search(query_embedding, 1)
    closest_objection_index = indices[0][0]
    return objections[closest_objection_index], responses[closest_objection_index]

print("Ready for real-time interaction. Speak into the microphone.")

try:
    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription = result.get("text", "")

            if transcription.strip():
                print(f"User: {transcription}")

                # Recommend products
                recommendations = recommend_products(transcription)
                print("\nProduct Recommendations:")
                for title, description in recommendations:
                    print(f"- {title}: {description}")

                # Handle objections
                objection, response = handle_objection(transcription)
                print("\nObjection Handling:")
                print(f"Objection: {objection}\nResponse: {response}")

                # Analyze sentiment
                sentiment = analyze_sentiment(transcription)
                print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']}")

                # Save to Google Sheets
                append_to_sheet(sentiment, transcription)
                print("Data saved to Google Sheets.")

except KeyboardInterrupt:
    print("\nExiting...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
