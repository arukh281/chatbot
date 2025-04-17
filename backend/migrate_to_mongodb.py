import json
import pymongo
from pymongo import MongoClient
import os
import re
import string
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["university_faq"]
faculty_collection = db["faculty"]
questions_collection = db["questions"]

# Simple tokenization function (no NLTK dependency)
def tokenize(text):
    if not isinstance(text, str):
        return []
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # Split by whitespace
    return [token.strip() for token in text.split() if token.strip()]

# Simple stemming function (for handling plurals without NLTK)
def simple_stem(word):
    """Very basic stemming for English words"""
    if not word or not isinstance(word, str):
        return word
        
    word = word.lower()
    
    # Common plural endings
    if len(word) > 3:
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'  # universities -> university
        elif word.endswith('es') and len(word) > 3:
            return word[:-2]  # classes -> class
        elif word.endswith('s') and not word.endswith('ss'):
            return word[:-1]  # students -> student
    
    return word

# Text preprocessing function (NLTK-free)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        
    # Tokenize
    tokens = tokenize(text)
    
    # Apply simple stemming
    stemmed_tokens = [simple_stem(token) for token in tokens]
    
    # Rejoin tokens
    return ' '.join(stemmed_tokens)

def extract_important_terms(text):
    """Extract important keywords from text that should be mandatory for matching."""
    if not isinstance(text, str):
        return []
    
    # List of common words that aren't important for matching
    common_words = {
        'where', 'what', 'when', 'how', 'who', 'why', 
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by',
        'about', 'like', 'through', 'over', 'before', 'between', 'after',
        'since', 'without', 'under', 'within', 'along', 'following',
        'can', 'could', 'should', 'would', 'may', 'might', 'must',
        'do', 'does', 'did', 'doing'
    }
    
    # Clean text
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    words = [w.strip() for w in text.split() if w.strip()]
    
    # Extract potentially important terms (not in common_words)
    important_terms = [w for w in words if w not in common_words]
    
    # Extract potential proper nouns (capitalized words in original text)
    original_words = text.split()
    proper_nouns = []
    for word in original_words:
        if word and word[0].isupper():
            proper_nouns.append(word.lower())
    
    # Prioritize proper nouns and multi-word terms
    priority_terms = list(set(proper_nouns))
    
    # Add any terms that might be locations or specific identifiers
    # (numbers, building names, etc.)
    specific_indicators = [
        w for w in important_terms 
        if any(char.isdigit() for char in w)  # Contains digits
        or len(w) > 3  # Longer terms are often more specific
    ]
    
    priority_terms.extend(specific_indicators)
    
    return list(set(priority_terms))

def migrate_faculty_data():
    # Clear existing faculty data
    faculty_collection.delete_many({})
    
    # Load faculty data from JSON file
    with open("faculty_details.json", "r") as f:
        faculty_data = json.load(f)
    
    # Insert faculty data into MongoDB
    for faculty in faculty_data:
        # Convert MongoDB Extended JSON format to standard format
        if "_id" in faculty:
            if isinstance(faculty["_id"], dict) and "$oid" in faculty["_id"]:
                faculty["_id"] = faculty["_id"]["$oid"]
        
        if "Mobile No" in faculty:
            if isinstance(faculty["Mobile No"], dict) and "$numberLong" in faculty["Mobile No"]:
                faculty["Mobile No"] = str(faculty["Mobile No"]["$numberLong"])
        
        faculty_collection.insert_one(faculty)
    
    print(f"Migrated {len(faculty_data)} faculty records to MongoDB.")

def migrate_questions_data():
    # Clear existing questions data
    questions_collection.delete_many({})
    
    # Load questions data from JSON file with UTF-8 encoding
    with open("chatbot_questions.json", "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # Insert questions data into MongoDB
    for question in questions_data:
        # Convert MongoDB Extended JSON format to standard format
        if "_id" in question:
            if isinstance(question["_id"], dict) and "$oid" in question["_id"]:
                question["_id"] = question["_id"]["$oid"]
        
        # Extract important terms for keyword matching
        important_terms = extract_important_terms(question.get("Question", ""))
        question["important_terms"] = important_terms
        questions_collection.insert_one(question)
    
    print(f"Migrated {len(questions_data)} questions to MongoDB.")

if __name__ == "__main__":
    # Migrate data
    migrate_faculty_data()
    migrate_questions_data()