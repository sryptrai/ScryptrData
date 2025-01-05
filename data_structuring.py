import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas_profiling import ProfileReport
from langdetect import detect
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from typing import List, Dict, Any


# Clean and normalize text
def clean_text(text: str) -> str:
    """
    Cleans input text by removing HTML tags, special characters, and leading/trailing spaces.
    Converts text to lowercase.
    """
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.strip().lower()
    return text


def remove_stopwords(text: str) -> str:
    """
    Removes stopwords from the given text using NLTK's English stopwords list.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])


# Normalize numerical data
def normalize_data(data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Normalizes specified numerical columns in a DataFrame using Min-Max Scaling.
    """
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data


# Flatten nested JSON
def flatten_json(json_obj: Dict, separator: str = '_') -> pd.DataFrame:
    """
    Flattens a nested JSON object into a flat DataFrame.
    """
    return pd.json_normalize(json_obj, sep=separator)


# Generate exploratory data analysis (EDA) reports
def generate_eda_report(data: pd.DataFrame, output_file: str = 'eda_report.html') -> None:
    """
    Generates an EDA report using pandas-profiling and saves it as an HTML file.
    """
    profile = ProfileReport(data, title="Exploratory Data Analysis Report")
    profile.to_file(output_file)


# Detect language
def detect_language(text: str) -> str:
    """
    Detects the language of the given text using the langdetect library.
    """
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "Unknown"


# Extract named entities
def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extracts named entities from the input text using SpaCy's small English model.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities


# Example usage of the module
if __name__ == "__main__":
    # Example raw JSON data
    raw_json = {
        "user": {
            "id": 123,
            "name": "John Doe",
            "details": {"email": "john@example.com", "age": 30}
        },
        "preferences": {"theme": "dark", "notifications": True}
    }

    # Flatten JSON
    flat_data = flatten_json(raw_json)
    print("Flattened Data:")
    print(flat_data)

    # Example text data
    text = "The Eiffel Tower is located in Paris, France."

    # Clean text
    cleaned_text = clean_text(text)
    print("Cleaned Text:", cleaned_text)

    # Remove stopwords
    text_without_stopwords = remove_stopwords(cleaned_text)
    print("Text without Stopwords:", text_without_stopwords)

    # Detect language
    language = detect_language(text)
    print("Detected Language:", language)

    # Extract entities
    entities = extract_entities(text)
    print("Extracted Entities:", entities)

