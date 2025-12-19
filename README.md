# AI_Agent_Educations
Study bot for computer science
Project worked on within the group EdatozTraining
Agentic Chat + Code Runner
Streamlit app with three components:
Agentic chatbot using Groq (Llama 3) and LangChain tools
Python code runner for uploaded .py or .zip projects
Quiz generator (MCQ, True/False, Short Answer) with JSON/CSV export

Setup
pip install -r requirements.txt

Create .env:
GROQ_API_KEY=your_key_here

Run:
streamlit run app.py

Notes

Automatically installs requirements.txt for uploaded projects
Detects entrypoint (main.py, app.py, __main__.py, etc.)
Executes uploaded code locally; not safe for public deployment
