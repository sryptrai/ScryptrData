# ScryptrData

ScryptrData is a sophisticated AI-powered application designed to scrape, structure, and provide real-time data for AI agents and applications. This FastAPI-based project enables seamless integration of web scraping, adaptive pipelines, and intelligent APIs for AI agent support.

## Features

- **AI-Ready Data Structuring**: Transform raw web data into structured formats optimized for training Large Language Models and AI agents.
- **Autonomous Agent Support**: Enable AI agents to autonomously gather real-time data from websites, APIs, and social platforms for informed decision-making.
- **Adaptive Learning Pipeline**: Continuous data collection and processing pipeline to help AI agents stay updated with real-world information.
- **Comprehensive APIs**: Provide endpoints for sentiment analysis, summarization, language detection, and more.

---

## Repository Structure

### 1. **`ai_scraping_module.py`**
This is the main application file that initializes the FastAPI server and contains the following key functionalities:
- **Static Content Scraping**: Extract content from static websites.
- **Dynamic Content Scraping**: Use Playwright to scrape dynamic content rendered by JavaScript.
- **API Data Fetching**: Query APIs and extract data asynchronously.
- **Sentiment Analysis**: Perform sentiment analysis using TextBlob.
- **Text Summarization**: Summarize long texts using Hugging Face's Transformers.
- **Language Detection**: Detect the language of input text.
- **Named Entity Recognition (NER)**: Extract entities like names, places, and organizations using SpaCy.
- **Normalization**: Normalize numerical data for analysis.
- **JSON Flattening**: Flatten nested JSON structures for easier processing.

### 2. **`data_structuring.py`**
Handles the transformation of raw data into structured formats, featuring:
- **Text Cleaning**: Remove HTML tags, special characters, and stop words.
- **Metadata Extraction**: Extract metadata like title and keywords from HTML pages.
- **Normalization**: Scale numerical data using MinMaxScaler for better comparability.
- **Table Extraction**: Extract tabular data from HTML content using BeautifulSoup and Pandas.

### 3. **`adaptive_pipeline.py`**
Implements a continuous adaptive pipeline for real-time data collection and processing:
- **Scheduler Integration**: Periodically scrape data using APScheduler.
- **Learning Pipeline**: Dynamically updates AI-ready data based on new inputs.
- **Real-Time Updates**: Processes data continuously to ensure freshness and relevance.

### 4. **`api.py`**
Provides intelligent APIs that can be accessed by AI agents for real-time insights:
- **Sentiment Analysis Endpoint**: Analyze the sentiment of user-provided text.
- **Summarization Endpoint**: Generate concise summaries of lengthy content.
- **Language Detection Endpoint**: Detect and return the language of input text.
- **Entity Extraction Endpoint**: Extract named entities from text for deeper understanding.



---

## Installation and Setup

### Prerequisites
- Python 3.8+
- MongoDB installed and running
- Redis installed and running

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sryptrai/ScryptrData.git
   cd ScryptrData

