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

# Check faculty data
print("Faculty Data Structure:")
faculty_sample = faculty_collection.find_one()
if faculty_sample:
    print(faculty_sample)
else:
    print("No faculty data found")

# Check questions data
print("\nQuestions Data Structure:")
questions_sample = questions_collection.find_one()
if questions_sample:
    print(questions_sample)
else:
    print("No questions data found")

# Count documents
faculty_count = faculty_collection.count_documents({})
questions_count = questions_collection.count_documents({})

print(f"\nTotal faculty records: {faculty_count}")
print(f"Total questions records: {questions_count}") 