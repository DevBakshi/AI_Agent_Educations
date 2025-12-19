import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

# Send a chat completion request
chat_completion = client.chat.completions.create(
    model="llama3-8b-8192",  # free Groq model
    messages=[
        {"role": "user", "content": "Hello, Groq! How are you?"}
    ],
)

# Print the response
print(chat_completion.choices[0].message.content)
