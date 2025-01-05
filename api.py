from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import pandas as pd
from pymongo import MongoClient
from redis import Redis

# Set up logging
logging.basicConfig(level=logging.INFO, filename="agent_api.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI(title="AI Agent Interaction API")

# MongoDB client setup
client = MongoClient("mongodb://localhost:27017/")
db = client["agent_data"]

# Redis for caching
redis_client = Redis(host='localhost', port=6379, decode_responses=True)

# Request Models
class FetchRequest(BaseModel):
    source: str
    filters: Dict[str, Any] = None

class ProcessRequest(BaseModel):
    text: str
    operations: List[str]

class CustomQueryRequest(BaseModel):
    query_type: str
    params: Dict[str, Any]

# Utility Functions
def get_cached_data(key: str) -> Any:
    """
    Retrieve data from Redis cache.
    """
    try:
        cached_data = redis_client.get(key)
        if cached_data:
            logging.info(f"Cache hit for key: {key}")
            return cached_data
        logging.info(f"Cache miss for key: {key}")
        return None
    except Exception as e:
        logging.error(f"Error accessing Redis cache: {e}")
        return None

def cache_data(key: str, data: Any, ttl: int = 3600):
    """
    Store data in Redis cache.
    """
    try:
        redis_client.set(key, data, ex=ttl)
        logging.info(f"Data cached with key: {key}")
    except Exception as e:
        logging.error(f"Error caching data: {e}")

def apply_operations(text: str, operations: List[str]) -> str:
    """
    Apply text processing operations like sentiment analysis, summarization, etc.
    """
    processed_text = text
    for operation in operations:
        if operation == "summarize":
            processed_text = summarize_text(processed_text)
        elif operation == "sentiment":
            processed_text = analyze_sentiment(processed_text)
        elif operation == "ner":
            processed_text = extract_entities(processed_text)
    return processed_text

# Example Processing Functions
def summarize_text(text: str) -> str:
    return f"Summarized: {text[:50]}..."  # Placeholder logic

def analyze_sentiment(text: str) -> str:
    return "Positive" if "good" in text.lower() else "Negative"

def extract_entities(text: str) -> List[Dict[str, str]]:
    return [{"entity": "Example", "type": "NOUN"}]  # Placeholder logic

# Endpoints
@app.get("/data/fetch")
async def fetch_data(request: FetchRequest):
    """
    Fetch data from MongoDB or Redis cache.
    """
    cache_key = f"fetch:{request.source}:{hash(str(request.filters))}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return {"message": "Data retrieved from cache", "data": cached_data}

    # Fetch from MongoDB
    try:
        collection = db[request.source]
        query = request.filters or {}
        data = list(collection.find(query))
        cache_data(cache_key, data)
        return {"message": "Data fetched successfully", "data": data}
    except Exception as e:
        logging.error(f"Error fetching data from {request.source}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch data")

@app.post("/data/process")
async def process_data(request: ProcessRequest):
    """
    Process text data with specified operations.
    """
    try:
        processed_result = apply_operations(request.text, request.operations)
        return {"message": "Data processed successfully", "result": processed_result}
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        raise HTTPException(status_code=500, detail="Data processing failed")

@app.post("/query/custom")
async def custom_query(request: CustomQueryRequest):
    """
    Perform a custom query based on agent needs.
    """
    try:
        if request.query_type == "aggregate":
            # Example: Custom aggregation query on MongoDB
            pipeline = request.params.get("pipeline", [])
            collection_name = request.params.get("collection")
            collection = db[collection_name]
            results = list(collection.aggregate(pipeline))
            return {"message": "Custom query executed successfully", "results": results}
        else:
            return {"message": "Query type not supported", "results": []}
    except Exception as e:
        logging.error(f"Error executing custom query: {e}")
        raise HTTPException(status_code=500, detail="Custom query failed")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for agents.
    """
    try:
        db_status = client.admin.command('ping')
        redis_status = redis_client.ping()
        return {"status": "OK", "db_status": db_status, "redis_status": redis_status}
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

