from response import chatbot_response

# Test questions
test_questions = [
    "What programs does the CSE department at MUJ offer?",
    "Who is the head of the CSE department at MUJ?",
    "amit garg",  # Just the name
    "Contact of Dr. Amit Garg",  # Explicit contact request
    "What is the contact information for Dr. Neha Chaudhary?",
    "Contact of Dr. Mahesh Jangid",
    "What are the placement statistics for CSE students at MUJ?"
]

# Test each question
for question in test_questions:
    print(f"\nQuestion: {question}")
    result = chatbot_response(question)
    
    # Handle different return formats
    if len(result) == 3:  # Confirmation case with actual answer
        confidence, response, actual_answer = result
        print(f"Confidence: {confidence}")
        print(f"Response: {response}")
        print(f"Actual Answer: {actual_answer}")
    else:  # Normal case
        confidence, response = result
        print(f"Confidence: {confidence}")
        print(f"Response: {response}") 