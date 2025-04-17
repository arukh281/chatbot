from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from response import chatbot_response  # Import your chatbot function
import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime  # Add this import

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS

# MongoDB connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["university_faq"]
feedback_collection = db["feedback"]
faculty_collection = db["faculty"]
questions_collection = db["questions"]

# Ensure collections exist
if "feedback" not in db.list_collection_names():
    db.create_collection("feedback")
if "faculty" not in db.list_collection_names():
    db.create_collection("faculty")
if "questions" not in db.list_collection_names():
    db.create_collection("questions")

def save_to_collection(collection_name, question, answer, feedback_type):
    collection = db[collection_name]
    document = {
        "question": question,
        "answer": answer,
        "feedback_type": feedback_type,
        "timestamp": datetime.now()  # Use datetime.now() instead of pymongo.datetime
    }
    collection.insert_one(document)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    result = chatbot_response(user_message)  # This now returns (confidence, response, actual_answer) or (confidence, response)

    # Check if result is a tuple with 3 elements (confirmation case with actual answer)
    if isinstance(result, tuple) and len(result) == 3:
        confidence, response, actual_answer = result
        return jsonify({
            'response': response,
            'confidence': confidence,
            'actualAnswer': actual_answer
        })
    # Check if result is a tuple with 2 elements (normal case)
    elif isinstance(result, tuple) and len(result) == 2:
        confidence, response = result
        return jsonify({
            'response': response,
            'confidence': confidence
        })
    # Fallback case
    else:
        return jsonify({
            'response': str(result),
            'confidence': 0.0
        })

@app.route('/feedback', methods=['POST'])
def save_feedback():
    try:
        data = request.json
        feedback = {
            'question': data.get('question', ''),
            'answer': data.get('answer', ''),
            'feedback_type': data.get('feedback_type', 'not_helpful'),
            'timestamp': datetime.now()
        }
        
        # Save to MongoDB
        feedback_collection.insert_one(feedback)
        
        return jsonify({'status': 'success', 'message': 'Feedback saved successfully'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "University FAQ Chatbot API is running. Send POST requests to /chat"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
