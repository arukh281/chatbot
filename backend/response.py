import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from fuzzywuzzy import fuzz
import re
import string
import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["university_faq"]
faculty_collection = db["faculty"]
questions_collection = db["questions"]

# Load faculty data from MongoDB
def get_faculty_data():
    faculty_data = {}
    for faculty in faculty_collection.find():
        name = faculty.get("Name of Faculty ", "")
        email = faculty.get("E-mail ID ", "")
        phone = faculty.get("Mobile No", "")
        seating = faculty.get("Seating", "")
        if name:
            faculty_data[name] = {
                "email": email, 
                "phone": phone,
                "seating": seating
            }
    return faculty_data

# Load questions data from MongoDB
def get_questions_data():
    questions_data = []
    for question in questions_collection.find():
        # Extract important terms if not already present
        important_terms = question.get("important_terms", [])
        if not important_terms:
            # If not in MongoDB, extract them from the question
            important_terms = extract_important_terms(question.get("Question", ""))
            
        questions_data.append({
            "question": question.get("Question", ""),
            "answer": question.get("Answers", ""),
            "category": question.get("Keyword", ""),
            "important_terms": important_terms
        })
    return questions_data

# Initialize data
faculty_contacts = get_faculty_data()
questions_data = get_questions_data()

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

# Function to generate variations
def generate_variations(text):
    if not isinstance(text, str):
        return [text] if text else [""]
        
    variations = [text]
    
    # Simple plural/singular handling
    words = text.split()
    for i, word in enumerate(words):
        # Try both stemmed and original forms
        stemmed = simple_stem(word)
        if stemmed != word:
            new_words = words.copy()
            new_words[i] = stemmed
            variations.append(' '.join(new_words))
    
    # Add common misspellings for university terms
    common_misspellings = {
        'university': ['univercity', 'univarsity', 'uni'],
        'admission': ['admision', 'admisssion'],
        'scholarship': ['scolarship', 'scholarshipp'],
        'registration': ['registraton', 'registeration', 'signup'],
        'course': ['cours', 'coarse', 'class'],
        'professor': ['professer', 'proffesor', 'prof'],
        'semester': ['semister', 'semestre', 'term'],
        'tuition': ['tution', 'tuishon', 'fees'],
        'degree': ['degre', 'diploma', 'qualification'],
        'dormitory': ['dorm', 'housing', 'residence'],
        'major': ['specialization', 'concentration', 'field'],
        'credit': ['cred', 'unit', 'point'],
        'exam': ['examination', 'test', 'assessment'],
        'financial': ['fiscal', 'money', 'monetary'],
        'transfer': ['xfer', 'switch', 'change'],
        'deadline': ['due date', 'cutoff', 'timeframe'],
        'application': ['app', 'apply', 'submission'],
        'schedule': ['timetable', 'calendar', 'agenda']
    }
    
    for word, misspellings in common_misspellings.items():
        if word in text:
            for misspelling in misspellings:
                variations.append(text.replace(word, misspelling))
                
    return variations

# Build domain dictionary for spelling correction
def build_domain_dictionary():
    domain_dictionary = set()
    
    # Add terms from questions
    for question in questions_data:
        if isinstance(question.get("question"), str):
            tokens = tokenize(question.get("question"))
            domain_dictionary.update(tokens)
    
    # Add faculty names
    for faculty_name in faculty_contacts.keys():
        if isinstance(faculty_name, str):
            tokens = tokenize(faculty_name)
            domain_dictionary.update(tokens)
    
    return domain_dictionary

# Initialize domain dictionary
domain_dictionary = build_domain_dictionary()

def correct_spelling(text):
    if not isinstance(text, str):
        return str(text) if text else ""
        
    try:
        corrected = str(TextBlob(text).correct())
        
        # Protect domain-specific terms from "correction"
        words = text.lower().split()
        corrected_words = corrected.lower().split()
        
        # Only replace words if length matches
        if len(words) == len(corrected_words):
            for i, (orig_word, corr_word) in enumerate(zip(words, corrected_words)):
                if orig_word in domain_dictionary and orig_word != corr_word:
                    corrected_words[i] = orig_word
            
            return ' '.join(corrected_words)
        return corrected
    except Exception as e:
        print(f"Spelling correction error: {e}")
        return text

def get_faculty_contact(user_input):
    # Check if the input is likely to be a name
    if not is_likely_name(user_input):
        return {"matched_faculty": None, "confidence": 0.0, "match_type": "not_name"}
    
    # Try to match the name
    matched_name, confidence = fuzzy_match_faculty_name(user_input)
    
    if matched_name:
        # Get faculty data from MongoDB
        faculty_data = get_faculty_data()
        faculty_info = faculty_data.get(matched_name, {})
        
        # Check if this is a contact request or just a name query
        is_contact_request = bool(re.match(r"(?:contact|email|phone|details|location)\s+of\s+", user_input.lower()))
        
        if is_contact_request:
            email = faculty_info.get("email", "Email not available")
            phone = faculty_info.get("phone", "Phone not available")
            seating = faculty_info.get("seating", "")
            
            response = f"Contact information for {matched_name}:\n"
            if seating:
                response += f"Location: {seating}\n"
            response += f"Email: {email}\n"
            response += f"Phone: {phone}"
        else:
            response = f"Would you like to know the contact information for {matched_name}?"
        
        return {
            "matched_faculty": matched_name,
            "confidence": confidence,
            "match_type": "faculty_contact" if is_contact_request else "faculty_name",
            "answer": response
        }
    
    return {"matched_faculty": None, "confidence": confidence, "match_type": "no_match"}


import traceback

def match_input(processed_input, original_input):
    try:
        # Check if it's a faculty contact query
        faculty_result = get_faculty_contact(original_input)
        if faculty_result["matched_faculty"]:
            return faculty_result

        # Create a DataFrame from the questions_data
        df = pd.DataFrame(questions_data)
        
        if df.empty:
            return {'answer': "I don't have any information to help with your question.", 'confidence': 0.0, 'match_type': 'empty', 'matched_question': ""}
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        
        # Fit and transform the questions
        question_vectors = vectorizer.fit_transform(df['question'].tolist())
        
        # Transform the user input
        input_vector = vectorizer.transform([processed_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vector, question_vectors).flatten()
        
        # Get top 3 matches
        top_indices = similarities.argsort()[-3:][::-1]
        top_scores = similarities[top_indices]
        
        # If the best match is below threshold, return low confidence
        if top_scores[0] < 0.1:
            return {'answer': "I'm not sure about that. Could you rephrase your question?", 'confidence': top_scores[0], 'match_type': 'low_confidence', 'matched_question': df.iloc[top_indices[0]]['question']}
        
        # Extract important terms from input
        important_terms = extract_important_terms(processed_input)
        
        # Check for keyword matches using the important_terms field from MongoDB
        keyword_matches = []
        for term in important_terms:
            for idx, row in df.iterrows():
                # Check if the term is in the question text
                if term.lower() in row['question'].lower():
                    keyword_matches.append((idx, 0.8))  # Boost confidence for keyword matches
                
                # Also check if the term is in the important_terms field
                if 'important_terms' in row and isinstance(row['important_terms'], list):
                    if term.lower() in [t.lower() for t in row['important_terms']]:
                        keyword_matches.append((idx, 0.9))  # Higher boost for matches in important_terms
        
        # If we have keyword matches, use the one with highest similarity score
        if keyword_matches:
            best_keyword_match = max(keyword_matches, key=lambda x: x[1])
            return {
                'answer': df.iloc[best_keyword_match[0]]['answer'],
                'confidence': best_keyword_match[1],
                'match_type': 'keyword',
                'matched_question': df.iloc[best_keyword_match[0]]['question']
            }
        
        # Otherwise, return the best match based on similarity
        return {
            'answer': df.iloc[top_indices[0]]['answer'],
            'confidence': top_scores[0],
            'match_type': 'similarity',
            'matched_question': df.iloc[top_indices[0]]['question']
        }
        
    except Exception as e:
        print(f"Error in match_input: {e}")
        import traceback
        traceback.print_exc()
        return {'answer': "I encountered an error processing your question. Please try again.", 'confidence': 0.0, 'match_type': 'error', 'matched_question': ""}
    
def process_input(user_input):
    # First check if it's a university location query
    university_location_keywords = [
        "location of manipal", 
        "address of manipal", 
        "where is manipal", 
        "manipal location", 
        "manipal address",
        "location of manipal university",
        "address of manipal university",
        "where is manipal university",
        "manipal university location",
        "manipal university address"
    ]
    
    if any(keyword in user_input.lower() for keyword in university_location_keywords):
        return {
            'answer': "Manipal University Jaipur is located at:\nDehmi Kalan, Near GVK Toll Plaza,\nJaipur-Ajmer Expressway,\nJaipur, Rajasthan 303007, India.",
            'confidence': 1.0,
            'match_type': 'university_location'
        }
    
    # Check if it's a faculty contact query
    contact_triggers = ["contact of", "contact", "email", "phone", "number", "mobile", "how to reach", "call", "cabin"]
    if any(trigger in user_input.lower() for trigger in contact_triggers):
        faculty_response = get_faculty_contact(user_input)
        if faculty_response["matched_faculty"]:  # Ensure faculty is found
            return faculty_response  

    # If not a contact request, proceed with standard question matching
    processed_input = preprocess_text(user_input)
    return match_input(processed_input, user_input)

def get_response(user_input, confidence_threshold=0.5):
    if not isinstance(user_input, str) or not user_input.strip():
        return "I need a question to help you.", 0.0, ""

    result = process_input(user_input)  
    if result['confidence'] >= confidence_threshold:
        # Instead of returning the answer directly, ask for confirmation
        return f"Do you mean: {result.get('matched_question', '')}?", 0.0, result.get('matched_question', ''), result['answer']

    # Try spelling correction only for non-faculty queries
    if result["match_type"] != "faculty_contact":
        corrected_input = correct_spelling(user_input)
        if corrected_input.lower() != user_input.lower():
            processed_corrected = preprocess_text(corrected_input)
            corrected_result = match_input(processed_corrected, corrected_input)
            if corrected_result['confidence'] > result['confidence']:
                return f"Do you mean: {corrected_result.get('matched_question', '')}?", 0.0, corrected_result.get('matched_question', ''), corrected_result['answer']

    # Try fuzzy matching as another fallback
    fuzzy_result = fuzzy_match(user_input)
    if fuzzy_result['confidence'] > result['confidence']:
        return f"Do you mean: {fuzzy_result.get('matched_question', '')}?", 0.0, fuzzy_result.get('matched_question', ''), fuzzy_result['answer']

    if result['confidence'] > 0.15:
        return f"Do you mean: {result.get('matched_question', '')}?", 0.0, result.get('matched_question', ''), result['answer']

    return "I'm not sure I understand. Could you rephrase your question?", 0.0, ""




def fuzzy_match(user_input):
    try:
        # Create a DataFrame from the questions_data
        df = pd.DataFrame(questions_data)
        
        if df.empty:
            return {'answer': "I don't have any information to help with your question.", 'confidence': 0.0, 'match_type': 'empty', 'matched_question': ""}
        
        # Preprocess the input
        processed_input = preprocess_text(user_input)
        
        # Calculate fuzzy match scores for each question
        scores = []
        for idx, row in df.iterrows():
            question = row['question']
            # Calculate ratio for the entire question
            ratio = fuzz.ratio(processed_input.lower(), question.lower()) / 100.0
            
            # Calculate partial ratio for substring matches
            partial_ratio = fuzz.partial_ratio(processed_input.lower(), question.lower()) / 100.0
            
            # Calculate token sort ratio for word order differences
            token_sort_ratio = fuzz.token_sort_ratio(processed_input.lower(), question.lower()) / 100.0
            
            # Take the maximum of all ratios
            max_ratio = max(ratio, partial_ratio, token_sort_ratio)
            
            scores.append((idx, max_ratio))
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get the best match
        best_idx, best_score = scores[0]
        
        # If the best score is too low, return a low confidence response
        if best_score < 0.3:
            return {'answer': "I'm not sure about that. Could you rephrase your question?", 'confidence': best_score, 'match_type': 'low_confidence', 'matched_question': df.iloc[best_idx]['question']}
        
        return {
            'answer': df.iloc[best_idx]['answer'],
            'confidence': best_score,
            'match_type': 'fuzzy',
            'matched_question': df.iloc[best_idx]['question']
        }
        
    except Exception as e:
        print(f"Error in fuzzy_match: {e}")
        import traceback
        traceback.print_exc()
        return {'answer': "I encountered an error processing your question. Please try again.", 'confidence': 0.0, 'match_type': 'error', 'matched_question': ""}

def fuzzy_match_faculty_name(name):
    if not name:
        return None, 0.0
    
    # Preprocess the name
    processed_name = preprocess_text(name)
    
    # Get faculty data from MongoDB
    faculty_data = get_faculty_data()
    
    if not faculty_data:
        return None, 0.0
    
    # Calculate fuzzy match scores for each faculty name
    scores = []
    for faculty_name in faculty_data.keys():
        # Calculate ratio for the entire name
        ratio = fuzz.ratio(processed_name.lower(), faculty_name.lower()) / 100.0
        
        # Calculate partial ratio for substring matches
        partial_ratio = fuzz.partial_ratio(processed_name.lower(), faculty_name.lower()) / 100.0
        
        # Calculate token sort ratio for word order differences
        token_sort_ratio = fuzz.token_sort_ratio(processed_name.lower(), faculty_name.lower()) / 100.0
        
        # Take the maximum of all ratios
        max_ratio = max(ratio, partial_ratio, token_sort_ratio)
        
        scores.append((faculty_name, max_ratio))
    
    # Sort by score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the best match
    best_name, best_score = scores[0]
    
    # If the best score is too low, return None
    if best_score < 0.3:
        return None, best_score
    
    return best_name, best_score

def is_likely_name(text):
    """Check if text is likely a person's name and return appropriate response"""
    # Remove any titles
    text = re.sub(r'^(Dr\.|Prof\.)\s*', '', text.strip())
    
    # Split into words
    words = text.split()
    
    # Must have 1-3 words
    if len(words) < 1 or len(words) > 3:
        return False
    
    # Common non-name words that might appear in queries
    non_name_words = {
        'admission', 'process', 'fee', 'structure', 'course', 'duration',
        'placement', 'statistics', 'faculty', 'information', 'infrastructure',
        'library', 'campus', 'hostel', 'mess', 'cafeteria', 'sports',
        'scholarship', 'exam', 'result', 'syllabus', 'lab', 'project',
        'internship', 'job', 'company', 'recruiter', 'salary', 'package'
    }
    
    # Check if any word is in the non-name words list
    if any(word.lower() in non_name_words for word in words):
        return False
    
    # Each word must:
    # - Be at least 2 characters long
    # - Contain only letters and possibly dots
    for word in words:
        if len(word) < 2:
            return False
        if not re.match(r'^[A-Za-z.]+$', word):
            return False
    
    return True

def chatbot_response(user_message):
    # First check if it's a university location query
    university_location_keywords = [
        "location of manipal", 
        "address of manipal", 
        "where is manipal", 
        "manipal location", 
        "manipal address",
        "location of manipal university",
        "address of manipal university",
        "where is manipal university",
        "manipal university location",
        "manipal university address"
    ]
    
    if any(keyword in user_message.lower() for keyword in university_location_keywords):
        return 0.0, "Manipal University Jaipur is located at:\nDehmi Kalan, Near GVK Toll Plaza,\nJaipur-Ajmer Expressway,\nJaipur, Rajasthan 303007, India."

    # Check if it's a contact query
    contact_query_match = re.match(r"(?:contact|email|phone|details|location)\s+of\s+(.+)", user_message.lower())
    if contact_query_match:
        name = contact_query_match[1].strip()
        
        # First try exact match with the given name
        for faculty_name, contact_info in faculty_contacts.items():
            if name.lower() in faculty_name.lower() or faculty_name.lower() in name.lower():
                # Instead of returning contact info directly, ask for confirmation
                return 0.0, f"Would you like to know the contact information for {faculty_name}?", f"Contact information for {faculty_name}:\n" + \
                    (f"Location: {contact_info.get('seating', '')}\n" if contact_info.get('seating') else '') + \
                    f"Email: {contact_info.get('email', 'Email not available')}\n" + \
                    f"Phone: {contact_info.get('phone', 'Phone not available')}"
        
        # If no exact match, try fuzzy matching
        fuzzy_match, score = fuzzy_match_faculty_name(name)
        if fuzzy_match:
            # Get faculty data for the fuzzy match
            faculty_data = get_faculty_data()
            faculty_info = faculty_data.get(fuzzy_match, {})
            
            # Ask for confirmation with the fuzzy matched name
            return 0.0, f"Would you like to know the contact information for {fuzzy_match}?", f"Contact information for {fuzzy_match}:\n" + \
                (f"Location: {faculty_info.get('seating', '')}\n" if faculty_info.get('seating') else '') + \
                f"Email: {faculty_info.get('email', 'Email not available')}\n" + \
                f"Phone: {faculty_info.get('phone', 'Phone not available')}"
        
        return 0.0, f"Currently no contact exists for {name}"

    # Check if it's just a name (and looks like a real name)
    if is_likely_name(user_message):
        name = user_message.strip()
        
        # First check if this exact name exists
        for faculty_name, contact_info in faculty_contacts.items():
            if name.lower() in faculty_name.lower() or faculty_name.lower() in name.lower():
                return 0.0, f"Would you like to know the contact information for {faculty_name}?"
        
        # If no exact match, try fuzzy matching
        fuzzy_match, score = fuzzy_match_faculty_name(name)
        if fuzzy_match:
            return 0.0, f"Would you like to know the contact information for {fuzzy_match}?"
        
        # If no match at all, return with low confidence to trigger the clarification
        return 0.0, f"Would you like to know the contact information for {name}?"

    # For all other queries (not names), use the regular response system
    result = get_response(user_message)
    
    # Unpack response based on length
    if len(result) == 4:  # Confirmation case with actual answer
        confirmation_text, confidence, matched_question, actual_answer = result
        return confidence, confirmation_text, actual_answer
    elif len(result) == 3:  # Normal case with matched question
        answer, confidence, matched_question = result
        return confidence, answer
    else:
        answer, confidence = result
        return confidence, answer