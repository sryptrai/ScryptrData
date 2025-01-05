from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
import aiohttp
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, filename="adaptive_pipeline.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# MongoDB client setup
client = MongoClient("mongodb://localhost:27017/")
db = client["adaptive_pipeline_data"]

# Adaptive pipeline configurations
class AdaptivePipeline:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.data_sources = []

    def add_data_source(self, name: str, url: str, type_: str, interval: int):
        """
        Adds a data source to the pipeline.

        :param name: Identifier for the data source.
        :param url: URL or API endpoint for data scraping.
        :param type_: Type of data source ('static', 'dynamic', 'api').
        :param interval: Scraping interval in minutes.
        """
        self.data_sources.append({
            "name": name,
            "url": url,
            "type": type_,
            "interval": interval
        })
        self.scheduler.add_job(
            self.scrape_and_process_data,
            'interval',
            minutes=interval,
            args=[name, url, type_]
        )

    async def scrape_static(self, url: str) -> str:
        """
        Asynchronously scrape static content from the provided URL.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    return await response.text()
        except Exception as e:
            logging.error(f"Error scraping static data from {url}: {e}")
            return ""

    async def scrape_dynamic(self, url: str) -> str:
        """
        Asynchronously scrape dynamic content using Playwright (placeholder implementation).
        """
        # Placeholder: Replace with Playwright integration for dynamic content scraping
        logging.info(f"Scraping dynamic content from {url} (placeholder implementation).")
        return f"Dynamic content from {url}"

    async def scrape_api(self, endpoint: str) -> Dict:
        """
        Asynchronously fetch data from an API endpoint.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, timeout=10) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logging.error(f"Error scraping API data from {endpoint}: {e}")
            return {}

    async def scrape_and_process_data(self, name: str, url: str, type_: str) -> None:
        """
        Scrapes data from the specified source and processes it.
        """
        logging.info(f"Starting data collection for {name} ({type_}).")
        data = None
        if type_ == "static":
            data = await self.scrape_static(url)
        elif type_ == "dynamic":
            data = await self.scrape_dynamic(url)
        elif type_ == "api":
            data = await self.scrape_api(url)

        if data:
            self.process_data(name, data)
            logging.info(f"Data collection completed for {name}.")

    def process_data(self, source_name: str, raw_data: Any) -> None:
        """
        Processes the raw scraped data and stores it in MongoDB.
        """
        # Example processing: Convert raw HTML to title (for static sources)
        if isinstance(raw_data, str):  # HTML content
            soup = BeautifulSoup(raw_data, "html.parser")
            processed_data = {"title": soup.title.string if soup.title else "No Title"}
        elif isinstance(raw_data, dict):  # JSON data
            processed_data = raw_data
        else:
            logging.warning(f"Unhandled data type from source {source_name}.")
            return

        # Save processed data to MongoDB
        try:
            collection = db[source_name]
            collection.insert_one({
                "timestamp": datetime.now(),
                "data": processed_data
            })
            logging.info(f"Processed data saved for {source_name}.")
        except Exception as e:
            logging.error(f"Error saving processed data for {source_name}: {e}")

    def start_pipeline(self):
        """
        Starts the adaptive pipeline scheduler.
        """
        self.scheduler.start()
        logging.info("Adaptive Learning Pipeline started.")

    def stop_pipeline(self):
        """
        Stops the adaptive pipeline scheduler.
        """
        self.scheduler.shutdown()
        logging.info("Adaptive Learning Pipeline stopped.")


# Example usage
if __name__ == "__main__":
    pipeline = AdaptivePipeline()

    # Adding data sources
    pipeline.add_data_source("example_static", "https://example.com", "static", 10)
    pipeline.add_data_source("example_dynamic", "https://dynamic.example.com", "dynamic", 15)
    pipeline.add_data_source("example_api", "https://api.example.com/data", "api", 20)

    # Start the adaptive pipeline
    pipeline.start_pipeline()

    try:
        # Keep the script running to allow scheduler to execute jobs
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        pipeline.stop_pipeline()

