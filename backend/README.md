# University FAQ Chatbot with MongoDB

This is a Flask-based chatbot application that uses MongoDB to store and retrieve faculty information and frequently asked questions.

## Prerequisites

- Python 3.8 or higher
- MongoDB 4.4 or higher
- pip (Python package manager)

## Setup

1. Install MongoDB:
   - Download and install MongoDB from [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
   - Start the MongoDB service

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   - Create a `.env` file in the backend directory
   - Add the following line (replace with your MongoDB URI if different):
     ```
     MONGO_URI=mongodb://localhost:27017/
     ```

4. Migrate data from JSON files to MongoDB:
   ```
   python migrate_to_mongodb.py
   ```

5. Start the Flask application:
   ```
   python app.py
   ```

The server will start on http://localhost:5001

## API Endpoints

- `GET /`: Health check endpoint
- `POST /chat`: Send a message to the chatbot
  - Request body: `{"message": "Your question here"}`
  - Response: `{"response": "Chatbot response", "confidence": 0.95}`
- `POST /feedback`: Submit feedback on a response
  - Request body: `{"question": "User question", "answer": "Chatbot answer", "feedback_type": "doubtful" or "not_helpful"}`
  - Response: `{"message": "Feedback saved successfully"}`

## MongoDB Collections

- `faculty`: Stores faculty contact information
- `questions`: Stores FAQ questions and answers
- `feedback`: Stores user feedback on responses

## Data Migration

The `migrate_to_mongodb.py` script migrates data from the following JSON files to MongoDB:
- `faculty_details.json`: Faculty contact information
- `chatbot_questions.json`: FAQ questions and answers

## Troubleshooting

- If you encounter connection issues with MongoDB, make sure the MongoDB service is running
- Check that the MongoDB URI in your environment variables is correct
- Ensure all required Python packages are installed correctly 