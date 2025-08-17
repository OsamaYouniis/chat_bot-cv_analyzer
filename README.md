# chat_bot-cv_analyzer
This project is a Streamlit-based recruitment assistant that evaluates and ranks job candidates for a Call Center Agent position.
It uses LLM-powered analysis to process candidate profiles (PDF resumes) and compare them against a job description.

The app outputs a professional recruiter-style evaluation for the top 3 candidates, including:

Match summary (short evaluation)

Key strengths

Final ranking (without scores, based on strengths and suitability)

🚀 Features

✅ Upload multiple candidate resumes (PDF)
✅ Extract text from resumes automatically
✅ Input custom job descriptions
✅ AI-driven candidate evaluation using structured prompts
✅ Outputs clear ranking of the top 3 candidates for a Call Center position
✅ Simple, clean Streamlit UI

🛠️ Tech Stack

Python 3.10+

Streamlit – UI framework

LangChain – LLM orchestration

PyPDF2 – PDF text extraction

[OpenAI / Gemini / LLM API] – for candidate analysis

📂 Project Structure
├── app.py                # Main Streamlit app  
├── requirements.txt      # Project dependencies  
├── prompts/              # Prompt templates for LLM  
│   └── call_center.txt   # Custom prompt for call center analysis  
├── uploads/              # Candidate PDF uploads  
├── outputs/              # Final ranked analysis reports  
└── README.md             # Project documentation  

⚡ Installation

Clone the repo

git clone https://github.com/your-username/call-center-candidate-ranking.git
cd call-center-candidate-ranking


Create virtual environment & install dependencies

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Set your API Key (for OpenAI or Gemini)

export OPENAI_API_KEY="your_api_key_here"


(Windows PowerShell)

$env:OPENAI_API_KEY="your_api_key_here"

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Enter the job description for the Call Center position.

Upload PDF resumes of candidates.

The app will analyze and show:

Candidate name

Match summary

Key strengths

Final ranking of top 3 candidates
