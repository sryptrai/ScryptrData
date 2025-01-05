from fastapi import FastAPI, HTTPException, BackgroundTasks
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler
from playwright.sync_api import sync_playwright
from transformers import pipeline
from textblob import TextBlob
import requests
import pandas as pd
import logging
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from pandas_profiling import ProfileReport
from langdetect import detect
import spacy
import asyncio
import aiohttp
import uvicorn
from pydantic import BaseModel
from redis import Redis

# Initialize FastAPI app
app = FastAPI(title="Enhanced AI Data Scraping and Structuring Module")

# Set up logging
logging.basicConfig(level=logging.INFO, filename="scraping.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# MongoDB client setup
client = MongoClient("mongodb://localhost:27017/")
db = client["scraping_data"]

# Redis for caching
redis_client = Redis(host='localhost', port=6379, decode_responses=True)

# User-Agent generator
def get_random_user_agent():
    return UserAgent().random

# Scrape static website content asynchronously
async def scrape_website_async(url):
    headers = {"User-Agent": get_random_user_agent()}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                return await response.text()
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return None

# Scrape dynamic content with Playwright
async def scrape_dynamic_content_async(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)
            content = page.content()
            browser.close()
            return content
    except Exception as e:
        logging.error(f"Error scraping dynamic content from {url}: {e}")
        return None

# Scrape data from APIs asynchronously
async def scrape_api_async(endpoint, query=None, variables=None, headers=None):
    try:
        async with aiohttp.ClientSession() as session:
            if query:
                response = await session.post(endpoint, json={"query": query, "variables": variables}, headers=headers)
            else:
                response = await session.get(endpoint, headers=headers, timeout=10)
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logging.error(f"Error fetching data from API {endpoint}: {e}")
        return None

# Extract tables from HTML
async def extract_tables_async(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = pd.read_html(str(soup))
        return tables  # List of DataFrames
    except Exception as e:
        logging.error(f"Error extracting tables: {e}")
        return []

# Clean and normalize text
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.strip().lower()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])

# Normalize numerical data
def normalize_data(data, feature_columns):
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data

# Flatten nested JSON
def flatten_json(json_obj, separator='_'):
    return pd.json_normalize(json_obj, sep=separator)

# Extract metadata from HTML
def extract_metadata(soup, metadata_tags):
    metadata = {}
    for tag in metadata_tags:
        meta = soup.find('meta', attrs={'name': tag})
        if meta:
            metadata[tag] = meta['content']
    return metadata

# Text summarization using Hugging Face Transformers
def summarize_text(text):
    summarizer = pipeline("summarization")
    return summarizer(text, max_length=50, min_length=25, do_sample=False)

# Sentiment analysis using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Language detection and transliteration
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        logging.error(f"Error detecting language: {e}")
        return None

# Named entity recognition
def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Save data to MongoDB
def save_to_mongodb(data, collection_name):
    try:
        collection = db[collection_name]
        collection.insert_many(data if isinstance(data, list) else [data])
        logging.info("Data saved to MongoDB successfully.")
    except Exception as e:
        logging.error(f"Error saving to MongoDB: {e}")

# Scheduler for periodic scraping
def scheduled_scraper():
    logging.info("Scheduled scraping started.")
    url = "https://example.com"  # Replace with target URL
    html_content = asyncio.run(scrape_website_async(url))
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        data = {"title": soup.title.string}  # Example extraction
        save_to_mongodb(data, "website_data")

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_scraper, 'interval', hours=1)
scheduler.start()

# Pydantic model for request validation
class ScrapeRequest(BaseModel):
    url: str
    metadata_tags: list = []

# FastAPI endpoints
@app.get("/scrape/static")
async def scrape_static_endpoint(request: ScrapeRequest):
    html_content = await scrape_website_async(request.url)
    if not html_content:
        raise HTTPException(status_code=500, detail="Failed to scrape website.")
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = extract_metadata(soup, request.metadata_tags) if request.metadata_tags else {}
    return {"message": "Static content scraped successfully", "metadata": metadata, "title": soup.title.string}

@app.get("/scrape/dynamic")
async def scrape_dynamic_endpoint(url: str):
    content = await scrape_dynamic_content_async(url)
    if not content:
        raise HTTPException(status_code=500, detail="Failed to scrape dynamic content.")
    return {"message": "Dynamic content scraped successfully", "data": content}

@app.post("/scrape/api")
async def scrape_api_endpoint(endpoint: str, query: str = None, variables: dict = None):
    data = await scrape_api_async(endpoint, query, variables)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch API data.")
    return {"message": "API data fetched successfully", "data": data}

@app.post("/analyze/sentiment")
async def sentiment_analysis_endpoint(text: str):
    sentiment = analyze_sentiment(text)
    return {"message": "Sentiment analyzed successfully", "sentiment": sentiment}

@app.post("/summarize")
async def summarize_endpoint(text: str):
    summary = summarize_text(text)
    return {"message": "Text summarized successfully", "summary": summary}

@app.post("/normalize")
async def normalize_endpoint(data: dict, feature_columns: list):
    df = pd.DataFrame(data)
    normalized_data = normalize_data(df, feature_columns)
    return {"message": "Data normalized successfully", "data": normalized_data.to_dict()}

@app.post("/flatten")
async def flatten_endpoint(json_obj: dict):
    flattened = flatten_json(json_obj)
    return {"message": "JSON flattened successfully", "data": flattened.to_dict(orient='records')}

@app.post("/language/detect")
async def detect_language_endpoint(text: str):
    language = detect_language(text)
    return {"message": "Language detected successfully", "language": language}

@app.post("/entities")
async def entities_endpoint(text: str):
    entities = extract_entities(text)
    return {"message": "Entities extracted successfully", "entities": entities}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
