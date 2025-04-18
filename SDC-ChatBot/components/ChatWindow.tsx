"use client";

import { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import Image from 'next/image';

// Add this function at the top of ChatWindow.tsx or import it from a utility file
async function sendMessageToChatbot(userMessage: string) {
  try {
    const response = await fetch("http://127.0.0.1:5001/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userMessage }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Chatbot response:", data); // Debug log

    return {
      response: data.response,
      confidence: data.confidence,
      actualAnswer: data.actualAnswer || null
    };
  } catch (error) {
    console.error("Error in sendMessageToChatbot:", error);
    return {
      response: "Sorry, there was an error communicating with the chatbot.",
      confidence: 0,
      actualAnswer: null
    };
  }
}

async function sendFeedback(question: string, answer: string, feedbackType: string) {
  try {
    console.log('Sending feedback:', { question, answer, feedbackType }); // Debug log
    const response = await fetch("http://127.0.0.1:5001/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        answer,
        feedback_type: feedbackType
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  } catch (error) {
    console.error("Error sending feedback:", error);
  }
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  type?: 'confirmation' | 'normal' | 'clarification' | 'thank_you';
  feedbackRequested?: boolean;
  actualAnswer?: string;
}

const QuickQueries = [
  "Admission Process",
  "Course Duration",
  "Placement Statistics",
  "Faculty Information",
  "Infrastructure"
];

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Add user's message first
    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const { response, confidence, actualAnswer } = await sendMessageToChatbot(input);
      console.log("Response from chatbot:", { response, confidence, actualAnswer }); // Debug log

      // Check if this is a direct contact information response
      const isContactInfo = response.includes("Contact information for") &&
        (response.includes("Email:") || response.includes("Phone:") || response.includes("Location:"));

      if (isContactInfo) {
        // This is a direct contact information response
        const botResponse: Message = {
          id: (Date.now() + 1).toString(),
          text: response,
          sender: 'bot',
          feedbackRequested: true
        };
        setMessages((prev) => [...prev, botResponse]);

        // Add a thank you message
        const thankYouMessage: Message = {
          id: (Date.now() + 2).toString(),
          text: "Thank you for your contribution! What else can I help you with?",
          sender: 'bot',
          type: 'thank_you'
        };
        setMessages((prev) => [...prev, thankYouMessage]);
      }
      // If we have an actual answer, it means we need confirmation
      else if (actualAnswer) {
        const confirmationMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: response,
          sender: 'bot',
          type: 'confirmation',
          actualAnswer: actualAnswer // Store the actual answer
        };
        setMessages((prev) => [...prev, confirmationMessage]);
      } else {
        // Regular response
        const botResponse: Message = {
          id: (Date.now() + 1).toString(),
          text: response,
          sender: 'bot',
          feedbackRequested: confidence > 0.6
        };
        setMessages((prev) => [...prev, botResponse]);
      }
    } catch (error) {
      console.error("Error communicating with the chatbot:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, something went wrong. Please try again later.",
        sender: 'bot',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickQuery = async (query: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      text: query,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const { response, confidence, actualAnswer } = await sendMessageToChatbot(query);

      // If we have an actual answer, it means we need confirmation
      if (actualAnswer) {
        const confirmationMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: response,
          sender: 'bot',
          type: 'confirmation',
          actualAnswer: actualAnswer // Store the actual answer
        };
        setMessages(prev => [...prev, confirmationMessage]);
      } else {
        // Regular response
        const botResponse: Message = {
          id: (Date.now() + 1).toString(),
          text: response,
          sender: 'bot',
          feedbackRequested: confidence > 0.6
        };
        setMessages(prev => [...prev, botResponse]);
      }
    } catch (error) {
      console.error("Error communicating with the chatbot:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, something went wrong. Please try again later.",
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfirmation = async (response: 'yes' | 'no', message: Message) => {
    console.log("Handling confirmation:", response, message); // Debug log

    const userMessage: Message = {
      id: Date.now().toString(),
      text: response === 'yes' ? "Yes, that's correct" : "No, that's not what I meant",
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // If user clicks No, save the question to MongoDB
    if (response === 'no') {
      try {
        // Get the original question from the messages array
        const originalQuestion = messages[messages.indexOf(message) - 1]?.text || '';

        // Send to MongoDB
        const feedbackResponse = await fetch('http://127.0.0.1:5001/feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: originalQuestion,
            answer: message.text,
            feedback_type: 'not_helpful'
          }),
        });

        if (!feedbackResponse.ok) {
          throw new Error('Failed to save feedback');
        }
      } catch (error) {
        console.error('Error saving feedback:', error);
      }
    }

    setTimeout(() => {
      if (response === 'yes') {
        if (message.actualAnswer) {
          // If user confirmed and we have an actual answer, show it
          console.log("Showing actual answer:", message.actualAnswer); // Debug log
          const botResponse: Message = {
            id: (Date.now() + 1).toString(),
            text: message.actualAnswer,
            sender: 'bot',
            feedbackRequested: true
          };
          setMessages(prev => [...prev, botResponse]);
        } else if (message.text.includes("Would you like to know the contact information for")) {
          // Extract faculty name from the confirmation message
          const facultyName = message.text.replace("Would you like to know the contact information for", "").trim().replace("?", "");

          // Send a new query to get the contact information
          sendMessageToChatbot(`contact of ${facultyName}`).then(({ response }) => {
            const contactMessage: Message = {
              id: (Date.now() + 1).toString(),
              text: response,
              sender: 'bot',
              feedbackRequested: true
            };
            setMessages(prev => [...prev, contactMessage]);
          }).catch(error => {
            console.error("Error getting faculty contact:", error);
            const errorMessage: Message = {
              id: (Date.now() + 1).toString(),
              text: "Sorry, I couldn't retrieve the contact information. Please try again.",
              sender: 'bot'
            };
            setMessages(prev => [...prev, errorMessage]);
          });
        }
      } else {
        const botResponse: Message = {
          id: (Date.now() + 1).toString(),
          text: "I apologize for the confusion. Could you please rephrase your question?",
          sender: 'bot'
        };
        setMessages(prev => [...prev, botResponse]);
      }
      setIsLoading(false);
    }, 1000);
  };

  const handleFeedback = async (response: 'yes' | 'no', questionMsg: Message, answerMsg: Message) => {
    // Only save feedback if it's a "not helpful" response
    if (response === 'no') {
      try {
        // Get the actual question from the messages array
        const actualQuestion = messages[messages.indexOf(questionMsg) - 1]?.text || questionMsg.text;

        const response = await fetch('http://127.0.0.1:5001/feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: actualQuestion,
            answer: answerMsg.text,
            feedback_type: 'not_helpful'
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to save feedback');
        }
      } catch (error) {
        console.error('Error saving feedback:', error);
      }
    }

    if (response === 'yes') {
      // Check if this is a faculty contact query
      const isFacultyQuery = answerMsg.text.includes("Would you like to know the contact information for");
      const isLocationQuery = answerMsg.text.includes("Did you mean contact information for");

      if (isFacultyQuery || isLocationQuery) {
        // Extract the faculty name from the message
        let facultyName = "";
        if (isFacultyQuery) {
          facultyName = answerMsg.text.replace("Would you like to know the contact information for", "").trim();
        } else if (isLocationQuery) {
          facultyName = answerMsg.text.replace("Did you mean contact information for", "").trim();
        }

        // Remove the question mark if present
        const cleanFacultyName = facultyName.endsWith("?") ? facultyName.slice(0, -1) : facultyName;

        // Send a new query to get the contact information
        try {
          // Use a direct query format that the backend will recognize
          const { response, confidence, actualAnswer } = await sendMessageToChatbot(`contact of ${cleanFacultyName}`);

          // Add the contact information as a new message
          const contactMessage: Message = {
            id: (Date.now() + 1).toString(),
            text: response,
            sender: 'bot',
            feedbackRequested: true
          };
          setMessages((prev) => [...prev, contactMessage]);

          // Add a thank you message
          const thankYouMessage: Message = {
            id: (Date.now() + 2).toString(),
            text: "Thank you for your contribution! What else can I help you with?",
            sender: 'bot',
            type: 'thank_you'
          };
          setMessages((prev) => [...prev, thankYouMessage]);
        } catch (error) {
          console.error("Error getting faculty contact:", error);
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            text: "Sorry, I couldn't retrieve the contact information. Please try again.",
            sender: 'bot'
          };
          setMessages((prev) => [...prev, errorMessage]);
        }
      } else {
        // Regular thank you message for non-faculty queries
        const thankYouMessage: Message = {
          id: Date.now().toString(),
          text: "Thank you for your contribution! What else can I help you with?",
          sender: 'bot',
          type: 'thank_you'
        };
        setMessages((prev) => [...prev, thankYouMessage]);
      }
    }
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-orange-50 to-yellow-50">
      {/* Header */}
      <div className="bg-white shadow-md p-4 flex items-center gap-4">
        <Image
          src="\images\muj logo this.png"
          alt="MUJ Logo"
          width={40}
          height={40}
          className="h-8 w-auto sm:h-10 sm:w-auto md:h-12 md:w-auto lg:h-14 lg:w-auto"
        />
        <Image
          src="\images\sdc-logo-black.webp"
          alt="SDC Logo"
          width={40} // Adjust this if needed
          height={40}
          className="h-8 w-auto sm:h-10 sm:w-auto md:h-12 md:w-auto lg:h-14 lg:w-auto"
        />

        <h1 className="text-sm sm:text-base md:text-lg lg:text-xl xl:text-2xl antialiased font-semibold font-serif tracking-wide text-black-600">MUJ CSE ASSISTANT</h1>
      </div>

      {/* Quick Queries */}
      <div className="p-4 flex flex-wrap gap-2 justify-center">
        {QuickQueries.map((query, index) => (
          <motion.button
            key={query}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => handleQuickQuery(query)}
            className="bg-white px-4 py-2 rounded-full shadow-md hover:shadow-lg transition-shadow duration-200 text-orange-600 border border-orange-200 text-sm md:text-base"
          >
            {query}
          </motion.button>
        ))}
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] sm:max-w-[75%] p-3 md:p-4 rounded-2xl ${message.sender === 'user'
                ? 'bg-orange-500 text-white'
                : 'bg-white shadow-md'
                }`}
            >
              <p className="text-sm md:text-base whitespace-pre-line">{message.text}</p>
              {message.type === 'confirmation' && (
                <div className="mt-2 flex gap-2">
                  <button
                    onClick={() => handleConfirmation('yes', message)}
                    className="bg-green-500 text-white px-3 py-1 rounded-full text-sm"
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => handleConfirmation('no', message)}
                    className="bg-red-500 text-white px-3 py-1 rounded-full text-sm"
                  >
                    No
                  </button>
                </div>
              )}
              {message.type === 'clarification' && (
                <div className="mt-2 space-x-2">
                  <button
                    onClick={() => handleConfirmation('yes', message)}
                    className="text-sm text-gray-500 hover:text-green-500"
                  >
                    👍 Yes
                  </button>
                  <button
                    onClick={() => handleConfirmation('no', message)}
                    className="text-sm text-gray-500 hover:text-red-500"
                  >
                    👎 No
                  </button>
                </div>
              )}
              {message.sender === 'bot' && !message.type && (  // Only show for bot messages that aren't special types
                <div className="mt-2 space-x-2">
                  {message.text.includes("Would you like to know the contact information for") ||
                    message.text.includes("Did you mean contact information for") ? (
                    <>
                      <button
                        onClick={() => handleFeedback('yes', messages[index - 1], message)}
                        className="text-sm text-gray-500 hover:text-green-500"
                      >
                        👍 Yes
                      </button>
                      <button
                        onClick={() => handleFeedback('no', messages[index - 1], message)}
                        className="text-sm text-gray-500 hover:text-red-500"
                      >
                        👎 No
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => handleFeedback('yes', messages[index - 1], message)}
                        className="text-sm text-gray-500 hover:text-green-500"
                      >
                        👍 Helpful
                      </button>
                      <button
                        onClick={() => handleFeedback('no', messages[index - 1], message)}
                        className="text-sm text-gray-500 hover:text-red-500"
                      >
                        👎 Not Helpful
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white p-4 rounded-2xl shadow-md">
              <Loader2 className="h-6 w-6 animate-spin text-orange-500" />
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white border-t">
        <div className="max-w-4xl mx-auto flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(e)}
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 rounded-full border border-orange-200 focus:outline-none focus:ring-2 focus:ring-orange-500 text-sm md:text-base"
          />
          <button
            onClick={handleSendMessage}
            className="bg-orange-500 text-white p-3 rounded-full hover:bg-orange-600 transition-colors"
          >
            <Send className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  );
}