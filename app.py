import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
import vosk
import pyaudio
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables from the .env file
api_key = os.getenv('API_KEY')
model_path=os.getenv('MODEL_PATH')
prod_path=os.getenv('PROD_PATH')
obj_path=os.getenv('OBJ_PATH')
# Load product and objection data
@st.cache_resource
def load_data():
    product_data = pd.read_csv(prod_path)
    objections_data = pd.read_csv(obj_path)
    return product_data, objections_data

product_data, objections_data = load_data()

product_descriptions = product_data['description'].tolist()
product_titles = product_data['title'].tolist()
objections = objections_data['objection'].tolist()
responses = objections_data['response'].tolist()

# Initialize models
@st.cache_resource
def initialize_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vosk_model = vosk.Model(model_path)
    return model, vosk_model

model, vosk_model = initialize_models()

# Create embeddings and FAISS indices
@st.cache_resource
def create_indices():
    product_embeddings = model.encode(product_descriptions)
    objection_embeddings = model.encode(objections)

    product_index = faiss.IndexFlatL2(product_embeddings.shape[1])
    product_index.add(product_embeddings)

    objection_index = faiss.IndexFlatL2(objection_embeddings.shape[1])
    objection_index.add(objection_embeddings)

    return product_index, objection_index

product_index, objection_index = create_indices()

# Initialize audio stream
def initialize_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
    return audio, stream, recognizer

# Hugging Face API for sentiment analysis
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
API_KEY = api_key
headers = {"Authorization": f"Bearer {API_KEY}"}

def analyze_sentiment(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        sentiments = result[0]
        if len(sentiments) > 0:
            best_sentiment = max(sentiments, key=lambda x: x['score'])
            return best_sentiment
        else:
            return {"label": "ERROR", "score": 0.0}
    return {"label": "ERROR", "score": 0.0}

# Google Sheets API setup
@st.cache_resource
def setup_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("sheet").sheet1
    return sheet

sheet = setup_google_sheets()

def append_to_sheet(sentiment, transcription):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, sentiment['label'], sentiment['score'], transcription])

def recommend_products(query):
    query_embedding = model.encode([query])
    distances, indices = product_index.search(query_embedding, 3)
    return [(product_titles[i], product_descriptions[i]) for i in indices[0]]

def handle_objection(query):
    query_embedding = model.encode([query])
    distances, indices = objection_index.search(query_embedding, 1)
    idx = indices[0][0]
    return objections[idx], responses[idx]

# Function to save session data to a JSON file
def save_session_data(session_data):
    with open("session_data.json", "w") as f:
        json.dump(session_data, f)

# Streamlit UI
st.title("Real-Time Product Recommendation & Sentiment Analysis")

session_data = {
    "interactions": [],
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}


if st.button("Stop Listening"):
    st.info("Processing session data...")

    # Load session data from the JSON file
    with open("session_data.json", "r") as f:
        session_data = json.load(f)
    
    # Import and analyze data using dashboard.py
    from dashboard import analyze_data  # Ensure dashboard.py is in the same directory
    analysis_results = analyze_data(session_data)
    
    # Initialize variables for the summary
    summary_data = []
    all_recommendations = []
    objection_summary = []

    # Process each interaction in the session data
    for i, interaction in enumerate(session_data["interactions"]):
        # Extract relevant data
        customer_transcription = interaction['transcription']
        sentiment_label = interaction['sentiment']['label']
        product_recommendations = [rec[1] for rec in interaction['product_recommendations']]
        objection = interaction.get('objection_handling', None)
        
        # Build the narrative for each interaction
        if i == 0:
            # Start of the call
            summary_data.append(f"When the call started, the customer was {sentiment_label}. ")
        else:
            # Progression of the conversation
            summary_data.append(f"Then, the customer's tone shifted to {sentiment_label}. ")
        
        # Add product recommendations to the summary
        summary_data.append(f"We provided recommendations: {', '.join(product_recommendations)}. ")
        
        # Add objection handling if applicable
        #if objection:
        #    summary_data.append(f"Objection: {objection['objection']}. Response: {objection['response']}. ")
        
        # Collect all recommendations and objections for analysis
        #all_recommendations.extend(product_recommendations)
        #if objection:
        #    objection_summary.append(f"Objection: {objection['objection']}, Response: {objection['response']}")

    # Combine the summary data into one long narrative
    narrative_summary = " ".join(summary_data)
    overall_sentiment = (
        "Overall sentiment trends are depicted in the Call Summary Table and Sentiment Trends graph. "
        "Explore Sentiment Predictions below to anticipate the customer's future interests."
    )

    # Debug: Verify narrative summary
    print("Narrative Summary:", narrative_summary)
    print("All Recommendations:", all_recommendations)
    print("Objection Summary:", objection_summary)

    # Load BART model and tokenizer for summarization
    from transformers import BartForConditionalGeneration, BartTokenizer

    def load_bart_model():
        model_name = "facebook/bart-large-cnn"  # Pre-trained BART model for summarization
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        return model, tokenizer

    model, tokenizer = load_bart_model()

    # Function to generate summary using BART
    def generate_summary(text, model, tokenizer):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    # Generate the post-call summary
    call_summary = generate_summary(narrative_summary, model, tokenizer)
    
    # Display the post-call summary
    st.subheader("Post-Call Summary")
    st.write(f"Session Timestamp: {analysis_results['timestamp']}")
    st.write("Debug Narrative Summary:", narrative_summary)
    # Display the summary table
    
    st.subheader("Call Summary Table")
    st.dataframe(analysis_results["summary_table"])

    # Display the sentiment trends chart
    st.subheader("Sentiment Trends")
    st.pyplot(analysis_results["sentiment_chart"])

    # Display product recommendation trends
    st.subheader("Top Product Recommendations")
    st.pyplot(analysis_results["recommendation_chart"])

    # Display sentiment predictions (if available)
    st.subheader("Sentiment Predictions")
    sentiment_predictions = analysis_results["sentiment_predictions"]
    if isinstance(sentiment_predictions, str):  # Handle insufficient data case
        st.write(sentiment_predictions)
    else:
        st.line_chart(sentiment_predictions)

    # Display the word cloud for transcription topics
    st.subheader("Transcription Word Cloud")
    st.pyplot(analysis_results["wordcloud"])

    # Display actionable recommendations
    st.subheader("Actionable Recommendations")
    for recommendation in analysis_results["actionable_recommendations"]:
        st.write(f"- {recommendation}")


if st.button("Start Listening"):
    # Initialize session data and save to JSON at the start
    session_data = {"interactions": [], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open("session_data.json", "w") as f:
        json.dump(session_data, f, indent=4)

    st.info("Listening... Speak into the microphone.")
    audio, stream, recognizer = initialize_audio()

    # Initialize a variable to track the previous sentiment
    previous_sentiment = None  # Add this line

    try:
        while True:
            data = stream.read(4000)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                transcription = result.get("text", "")

                if transcription.strip():
                    st.write(f"User: {transcription}")

                    # Product Recommendations
                    recommendations = recommend_products(transcription)
                    st.subheader("Product Recommendations")
                    for title, description in recommendations:
                        st.write(f"- **{title}**: {description}")

                    # Objection Handling
                    objection, response = handle_objection(transcription)
                    st.subheader("Objection Handling")
                    st.write(f"**Objection:** {objection}")
                    st.write(f"**Response:** {response}")

                    # Sentiment Analysis
                    sentiment = analyze_sentiment(transcription)
                    st.subheader("Sentiment Analysis")
                    st.write(f"**Sentiment:** {sentiment['label']}")
                    st.write(f"**Score:** {sentiment['score']}")

                    # Track sentiment changes
                    if previous_sentiment and previous_sentiment != sentiment['label']:
                        st.warning(f"Sentiment changed from **{previous_sentiment}** to **{sentiment['label']}**.")  # Add this line
                    previous_sentiment = sentiment['label']  # Update previous sentiment

                    # Save to Google Sheets
                    append_to_sheet(sentiment, transcription)
                    st.success("Data saved to Google Sheets.")

                    # Add interaction to session data and update JSON file
                    interaction = {
                        "transcription": transcription,
                        "sentiment": sentiment,
                        "product_recommendations": recommendations,
                        "objection_handling": {"objection": objection, "response": response},
                    }
                    session_data["interactions"].append(interaction)

                    # Update JSON file after every interaction
                    with open("session_data.json", "w") as f:
                        json.dump(session_data, f, indent=4)

    except KeyboardInterrupt:
        st.warning("Stopped listening.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        st.success("Final session data saved to session_data.json.")
